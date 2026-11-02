"""Queued source calibration and real-mask study; never opens final confirmation."""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from elresearch.common import ROOT, digest, sha256, now, read_json, save_json
from elresearch.publication_dense_train import FILES, scientific_hashes, train
from elresearch.publication_dense_evaluate import evaluate_development
from elresearch.publication_splits import partition_manifest, fold_records


BASE = ROOT / 'artifacts/publication_20261017'
EXTRA = ['publication_dense_evaluate.py', 'publication_splits.py']


def core_specs():
    return [
        {'name': 'deeplabv3plus_full', 'architecture': 'deeplabv3plus_resnet50', 'adaptation': 'full', 'reference': False},
        {'name': 'dinov3_dense_frozen', 'architecture': 'dinov3_multidepth', 'adaptation': 'frozen', 'reference': False},
        {'name': 'dinov3_dense_lora', 'architecture': 'dinov3_multidepth', 'adaptation': 'lora', 'reference': False},
        {'name': 'dinov3_dense_full', 'architecture': 'dinov3_multidepth', 'adaptation': 'full', 'reference': False}]


def design():
    ucf_path = ROOT / 'data/ucf/manifest.json'
    ucf = read_json(ucf_path)
    train_ids = [r['id'] for r in ucf['records'] if r['partition'] == 'train']
    val_ids = [r['id'] for r in ucf['records'] if r['partition'] == 'validation']
    if len(train_ids) != 430 or len(val_ids) != 144: raise ValueError('UCF calibration scope changed')
    candidates = []
    rates = {'full': [1e-5, 1e-4], 'lora': [1e-4, 1e-3], 'frozen': [3e-4, 1e-3]}
    for spec in core_specs():
        values = [1e-4, 3e-4] if spec['architecture'].startswith('deeplab') else rates[spec['adaptation']]
        for value in values:
            candidates.append({**spec, 'family': spec['name'], 'name': spec['name'] + f'_lr{value:g}',
                               'encoder_lr': 0. if spec['adaptation'] == 'frozen' else value,
                               'head_lr': value if spec['adaptation'] == 'frozen' else 1e-3,
                               'geometry_weight': .1, 'top_fraction': .05})
    return {'version': 1, 'study_kind': 'Matched source pixel-supervision and normal-structure ablations',
        'registered_after': 'First low-label image-only source validation analysis; before any new dense SCDD model results or final confirmation scores',
        'split_manifest': partition_manifest(), 'seeds': [11, 22, 33],
        'calibration': {'dataset': 'ucf', 'manifest_sha256': sha256(ucf_path), 'training_ids': train_ids,
                        'validation_ids': val_ids, 'candidates': candidates, 'planned_runs': 24,
                        'selection': 'highest mean full-native source-validation pixel AP over three seeds after per-run checkpoint selection on fixed sampled pixels; earliest listed candidate on exact ties',
                        'limits': 'Reused source-only UCF development labels; provenance remains conditional. No SCDD or target labels select learning rates.'},
        'input_size': 336, 'batch_size': 2, 'updates_per_round': 50, 'maximum_rounds': 40, 'patience': 6,
        'weight_decay': 1e-4, 'target_source_recall': .95, 'device': 'auto',
        'selection': 'pooled pixel AP on up to 8192 fixed uniform known transformed pixels per source validation image; seed derived from image ID; earliest round on ties. Full native-mask metrics reported at selected checkpoint.',
        'preprocessing': 'Whole-field letterbox, native grayscale copied to RGB, ImageNet normalization; cracks max-preserved in downsized training targets; D4 synchronized across image, mask, validity, geometry and reference orientation.',
        'loss': 'image BCE + source-frequency inverse-square-root weighted 3-class pixel CE + 0.5 crack Dice; identical primary loss, no auxiliary supervision',
        'backbone_policy': 'DINOv3 stochastic layers deterministic; CNN pretrained BN statistics frozen, affine trained. New CNN ASPP/decoder uses GroupNorm. Different pretraining prevents an architecture-only causal claim.',
        'reference_policy': 'Only selected training masks, at most 256 images by frozen-CLS farthest-first coverage; same D4 orientation; exclude entire query capture family; retrieve 5 frames then normal patch match. Reference with live adapted encoder prohibited.',
        'planned_real_domain_runs': 144,
        'real_domain_schedule': {'internal_core_plus_reference': '5 systems x 4 budgets (5/10/20/100%) x 3 seeds = 60',
            'internal_reference_ablations': 'no-position and mean-local-evidence x 20/100% x 3 seeds = 12',
            'held_source_development': 'CNN, frozen DINO, LoRA, normal-reference x 3 sources x 20/100% x 3 seeds = 72'},
        'annotation_accounting': 'Selected training pixel masks AND all source validation pixel masks are counted; not comparable annotation cost to image-only experiments. UCF calibration labels additional.',
        'primary_readouts': ['native pooled crack pixel AP', 'positive-image Dice/IoU at fixed pixel probability .5',
                             'crack-negative image FPR and annotated-normal-structure pixel FPR', 'image AUROC/AP and recall/review at source-frozen threshold'],
        'limits': ['All dense source and held-source results are development evidence.',
                   'Fixed uniform validation-pixel sampling is for checkpoint selection only; final reported localization is full native scale.',
                   'The adapted DeepLabv3+ is not a reproduction of PV-S3 dual-teacher semi-supervision.',
                   'DINOv2 disabled; no course test, SCDD117, UCF507 or PVEL confirmation access.'],
        'disk_policy': 'Pause via explicit resource error before a new fit if free space <3 GiB; retain best checkpoints, delete only this run own completed latest-resume checkpoint.',
        'scientific_hashes': scientific_hashes(),
        'extra_scientific_hashes': {name: sha256(ROOT / 'elresearch' / name) for name in EXTRA},
        'runner_sha256': sha256(__file__), 'image_verifier_sha256': sha256(ROOT / 'verify_publication_runs.py')}


def register(folder):
    proposal = design(); proposal['study_id'] = digest(proposal)[:20]
    path = folder / 'protocol.json'
    if path.exists():
        old = read_json(path)
        if digest({k: v for k, v in old.items() if k != 'registered_at'}) != digest(proposal):
            raise RuntimeError('Dense registration changed; preserve protocol and use an explicit revision')
        return old
    snapshot = folder / 'source_snapshot'; snapshot.mkdir(parents=True, exist_ok=True)
    for name in FILES + EXTRA: shutil.copy2(ROOT / 'elresearch' / name, snapshot / name)
    shutil.copy2(__file__, snapshot / 'run_publication_dense_study.py')
    proposal['registered_at'] = now(); save_json(path, proposal)
    return proposal


def wait_for_image_study(folder):
    while True:
        status = read_json(BASE / 'scdd_development_v1/status.json')
        if status['state'] == 'complete':
            # Status may be written just before the prior runner releases its lock.
            with (BASE / 'scdd_development_v1/runner.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            return
        if status['state'] in {'failed', 'bounded_stop'}:
            raise RuntimeError('Image study needs repair/resume before queued dense work: ' + status['state'])
        save_json(folder / 'status.json', {'state': 'queued_after_image_study', 'updated_at': now(), 'pid': os.getpid(),
                                         'image_runs_complete': status['completed_runs'], 'image_runs_planned': status['planned_runs']})
        time.sleep(30)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--register-only', action='store_true')
    p.add_argument('--wait-for-image-study', action='store_true'); p.add_argument('--maximum-new-runs', type=int)
    args = p.parse_args(); folder = BASE / 'dense_development_v1'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'runner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = register(folder)
    print('DENSE REGISTERED', settings['study_id'], '24 calibration + 144 real-mask development fits', flush=True)
    if args.register_only: return
    completed, new = 0, 0
    def execute(spec, tr, va, seed, budget, destination, fold, target=None):
        nonlocal completed, new
        if not (destination / 'run.json').exists() and shutil.disk_usage(ROOT).free < 3 * 1024**3:
            raise RuntimeError('Less than 3 GiB free; preserve outputs and resolve storage before resuming')
        existed = (destination / 'run.json').exists()
        save_json(folder / 'status.json', {'state': 'running', 'updated_at': now(), 'pid': os.getpid(),
            'completed_runs': completed, 'planned_runs': 168,
            'current': {'fold': fold, 'budget': budget, 'seed': seed, 'model': spec['name']}})
        result = train(settings, spec, tr, va, seed, budget, destination)
        if target: evaluate_development({**settings, 'permitted_target_development_ids': [r['id'] for r in target]}, destination, target)
        completed += 1; new += int(not existed)
        if args.maximum_new_runs and new >= args.maximum_new_runs:
            raise StopIteration('Requested bounded run limit reached')
        return result
    try:
        if args.wait_for_image_study: wait_for_image_study(folder)
        elif read_json(BASE / 'scdd_development_v1/status.json')['state'] != 'complete':
            raise RuntimeError('Image GPU study still active; use --wait-for-image-study')
        if not (folder / 'image_checkpoint_verification_complete.json').exists():
            for stage in ['calibration', 'scdd']:
                subprocess.run([sys.executable, str(ROOT / 'verify_publication_runs.py'), '--stage', stage], cwd=ROOT, check=True)
            save_json(folder / 'image_checkpoint_verification_complete.json', {'at': now(), 'verifier_sha256': settings['image_verifier_sha256']})
        source = read_json(ROOT / 'data/ucf/manifest.json')['records']
        by_id = {r['id']: r for r in source}
        tr = [by_id[i] for i in settings['calibration']['training_ids']]
        va = [by_id[i] for i in settings['calibration']['validation_ids']]
        calibration_results = []
        for candidate in settings['calibration']['candidates']:
            for seed in settings['seeds']:
                result = execute(candidate, tr, va, seed, 1., folder / 'calibration' / f"{candidate['name']}_seed{seed}", 'ucf_calibration')
                calibration_results.append(result)
        chosen = []
        for base in core_specs():
            candidates = [s for s in settings['calibration']['candidates'] if s['family'] == base['name']]
            winner = max(candidates, key=lambda s: np.mean([r['validation_localization']['pooled_pixel_average_precision'] for r in calibration_results if r['model']['name'] == s['name']]))
            # Hyperparameter selection uses full native source-validation AP after
            # each candidate checkpoint was selected with fixed sampled pixels.
            chosen.append({**winner, 'name': base['name']})
        selection = {'at': now(), 'protocol_sha256': sha256(folder / 'protocol.json'), 'selected': chosen,
                     'rule': 'mean full-native UCF source-validation pixel AP, three seeds; earliest listed on ties'}
        selection_path = folder / 'selected_hyperparameters.json'
        if selection_path.exists():
            if read_json(selection_path)['selected'] != chosen: raise RuntimeError('Dense hyperparameter selection changed')
        else: save_json(selection_path, selection)
        frozen = next(x for x in chosen if x['adaptation'] == 'frozen')
        reference = {**frozen, 'name': 'dinov3_dense_reference', 'reference': True}
        no_position = {**reference, 'name': 'dinov3_reference_no_position', 'geometry_weight': 0.}
        mean = {**reference, 'name': 'dinov3_reference_mean_evidence', 'top_fraction': 1.}
        for fold in settings['split_manifest']['folds']:
            tr, va, target = fold_records(settings['split_manifest'], fold['name'])
            systems = chosen + [reference] if fold['name'] == 'internal' else [s for s in chosen + [reference] if s['name'] != 'dinov3_dense_full']
            for budget in fold['budgets']:
                specs = systems + ([no_position, mean] if fold['name'] == 'internal' and budget in [.2, 1.] else [])
                for seed in settings['seeds']:
                    for spec in specs:
                        destination = folder / 'runs' / fold['name'] / f'budget_{budget:g}' / f"{spec['name']}_seed{seed}"
                        execute(spec, tr, va, seed, budget, destination, fold['name'], target)
        if completed != 168: raise RuntimeError(f'Dense planned-run count mismatch: {completed}')
        save_json(folder / 'status.json', {'state': 'complete', 'updated_at': now(), 'pid': os.getpid(),
                                         'completed_runs': completed, 'planned_runs': 168,
                                         'next': 'Analyze development; final confirmation remains sealed'})
    except StopIteration as error:
        save_json(folder / 'status.json', {'state': 'bounded_stop', 'updated_at': now(), 'completed_runs': completed, 'reason': str(error)})
    except BaseException as error:
        save_json(folder / 'status.json', {'state': 'failed', 'updated_at': now(), 'completed_runs': completed,
                                         'error': str(error), 'traceback': traceback.format_exc()}); raise


if __name__ == '__main__': main()
