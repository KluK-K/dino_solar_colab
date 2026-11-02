"""Freeze scientific settings before opening source test or an external domain."""
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .common import ROOT, config, digest, read_json, save_json, sha256, now
from .data import records, load_mask, IGNORE
from .encoders import FrozenEncoder, extract, vectors, reproject
from .heads import NormalBank, InspectionHead, prepare_rows, head_predictions
from .probes import hog_stats
from .metrics import classification
from .anomaly import anomaly_score

def scientific_hashes():
    return {name: sha256(ROOT / 'elresearch' / name) for name in ['data.py', 'encoders.py', 'probes.py', 'heads.py', 'metrics.py', 'anomaly.py', 'evaluate.py']}

def freeze(run_id):
    folder = ROOT / 'artifacts/runs' / run_id
    run = read_json(folder / 'run.json')
    if run['dataset'] == 'elpv':
        raise ValueError('Research evaluation cannot open the ELPV course test')
    frozen = {'run_sha256': sha256(folder / 'run.json'), 'scientific_hashes': scientific_hashes(),
              'config_sha256': sha256(ROOT / 'config.json'), 'created_at': now(),
              'source_manifest': run['source_manifest'], 'publication_eligible': run['publication_eligible']}
    destination = folder / 'frozen.json'
    if destination.exists() and read_json(destination)['run_sha256'] != frozen['run_sha256']:
        raise RuntimeError('Cannot replace a frozen run')
    if not destination.exists():
        save_json(destination, frozen)
    return frozen

def validate_frozen(run_id):
    folder = ROOT / 'artifacts/runs' / run_id
    frozen = read_json(folder / 'frozen.json'); run = read_json(folder / 'run.json')
    if frozen['run_sha256'] != sha256(folder / 'run.json') or frozen['scientific_hashes'] != scientific_hashes() or frozen['config_sha256'] != sha256(ROOT / 'config.json'):
        raise RuntimeError('Frozen scientific settings or code changed')
    manifest = read_json(ROOT / 'data' / run['dataset'] / 'manifest.json')
    if manifest['signature'] != run['source_manifest']:
        raise RuntimeError('Source manifest changed')
    checkpoint = folder / {'probe': 'classifier.joblib', 'head': 'head.pt', 'anomaly': 'normal_bank.pt'}[run['kind']]
    if sha256(checkpoint) != run['checkpoint_sha256']:
        raise RuntimeError('Frozen checkpoint changed')
    if run.get('normal_bank_sha256') and sha256(folder / 'normal_bank.pt') != run['normal_bank_sha256']:
        raise RuntimeError('Frozen reference bank changed')
    return folder, run

def evaluate(run_id, dataset=None, model_path=None, allow_provisional=False):
    folder, run = validate_frozen(run_id)
    dataset = dataset or run['dataset']
    if dataset == 'elpv':
        raise PermissionError('ELPV research evaluation is disabled; original course evaluator owns its test')
    if not run['publication_eligible'] and not allow_provisional:
        raise PermissionError('Source provenance is provisional; only explicitly marked pilot evaluation is permitted')
    partition = 'test' if dataset == run['dataset'] else 'external'
    rows = records(dataset, partition, evaluation=True)
    source = read_json(ROOT / 'data' / run['dataset'] / 'manifest.json')
    if dataset != run['dataset']:
        source_pixels = {r['pixel_sha256'] for r in source['records']}
        source_ids = {r['id'] for r in source['records']}
        if any(r['pixel_sha256'] in source_pixels or r['id'] in source_ids for r in rows):
            raise ValueError('Cross-domain overlap detected; quarantine overlapping origins before freezing')
    ledger = folder / f'evaluation_{dataset}.json'
    if ledger.exists():
        raise RuntimeError('This evaluation domain was already opened; preserve its ledger and results')
    target = read_json(ROOT / 'data' / dataset / 'manifest.json')
    # Check encoder access and identity before marking target pixels as opened.
    started = time.perf_counter()
    encoder = None
    if run['encoder']['name'] != 'hog_stats':
        encoder = FrozenEncoder(run['encoder']['name'], run['encoder']['input_size'], config()['device'], model_path)
        if encoder.key != run['encoder_key']:
            raise RuntimeError('Evaluator loaded a different encoder or preprocessing')
    save_json(ledger, {'opened_at': now(), 'dataset': dataset, 'target_manifest': target['signature'],
                       'frozen_run': sha256(folder / 'frozen.json'), 'state': 'opened',
                       'selection_on_target': False})
    if run['encoder']['name'] == 'hog_stats':
        from .data import load_image
        X = np.array([hog_stats(load_image(r)) for r in rows])
        model = joblib.load(folder / 'classifier.joblib'); score = model.decision_function(X)
        maps = None
    else:
        extract(dataset, encoder, partition, config()['batch_size'], evaluation=True)
        if run['kind'] == 'probe':
            X = vectors(encoder, rows, run['representation'])
            model = joblib.load(folder / 'classifier.joblib'); score = model.decision_function(X)
            maps = None
        elif run['kind'] == 'anomaly':
            from .encoders import read_feature
            bank = NormalBank.load(folder / 'normal_bank.pt')
            features = [read_feature(encoder, row) for row in rows]
            outputs = [anomaly_score(f, bank, row['split_group']) for f, row in zip(features, rows)]
            score = np.array([out[0] for out in outputs]); maps = [out[1] for out in outputs]
            items = [{'geometry': f['geometry']} for f in features]
        else:
            bank = NormalBank.load(folder / 'normal_bank.pt') if run['method'] == 'reference' else None
            items = prepare_rows(rows, encoder, bank, run['method'], run['geometry_weight'], encoder.device)
            model = InspectionHead(run['dimensions'], run['config']['hidden_dim'], run['config']['top_fraction']).to(encoder.device)
            model.load_state_dict(torch.load(folder / 'head.pt', map_location=encoder.device, weights_only=True))
            score, maps = head_predictions(model, items, encoder.device, config()['batch_size'])
    y = np.array([r['target'] for r in rows])
    metric = classification(y, score, run['threshold'])
    result = {'run_id': run_id, 'dataset': dataset, 'partition': partition, 'metrics': metric,
              'threshold_origin': 'source_validation', 'threshold_refitted_on_target': False,
              'seconds_including_encoder_load_cache_or_extraction_and_scoring': time.perf_counter() - started,
              'publication_eligible': bool(run['publication_eligible'] and target.get('publication_eligible')),
              'scope': 'confirmed protocol' if run['publication_eligible'] else 'provenance-unverified pilot',
              'group_semantics': target.get('group_semantics'), 'completed_at': now()}
    if maps is not None:
        aps, dice, iou, pixel_truth, pixel_scores = [], [], [], [], []
        for row, item, values in zip(rows, items, maps):
            width = item['geometry']['size'] // encoder.patch_size
            predicted = reproject(values.reshape(width, width), item['geometry'])
            mask = load_mask(row); valid = mask != IGNORE; truth = mask == 1
            pixel_truth.append(truth[valid]); pixel_scores.append(predicted[valid])
            if truth[valid].any():
                aps.append(float(average_precision_score(truth[valid], predicted[valid])))
            hard = (predicted > .5) & valid; true = truth & valid
            intersection = int((hard & true).sum()); total = int(hard.sum() + true.sum()); union = int((hard | true).sum())
            dice.append(2 * intersection / total if total else 1.)
            iou.append(intersection / union if union else 1.)
            destination = folder / 'maps' / dataset / (digest(row['id']) + '.npz')
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, score=predicted.astype(np.float16), truth=mask)
        result['localization'] = {'mean_pixel_AP_on_positive_images': float(np.mean(aps)) if aps else None,
                                  'pooled_pixel_AP': float(average_precision_score(np.concatenate(pixel_truth), np.concatenate(pixel_scores))) if any(t.any() for t in pixel_truth) else None,
                                  'positive_images_for_AP': len(aps), 'mean_Dice_all_images': float(np.mean(dice)),
                                  'mean_IoU_all_images': float(np.mean(iou)), 'pixel_threshold': .5,
                                  'note': 'Native annotation regions; fixed pixel threshold; positive-image macro AP is not pooled pixel AP.'}
    np.savez_compressed(folder / f'predictions_{dataset}.npz', ids=np.array([r['id'] for r in rows]), y=y, scores=score,
                        groups=np.array([r['split_group'] for r in rows]))
    save_json(folder / f'results_{dataset}.json', result)
    save_json(ledger, {**read_json(ledger), 'state': 'completed', 'results_sha256': sha256(folder / f'results_{dataset}.json')})
    return result
