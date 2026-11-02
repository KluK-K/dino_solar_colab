"""Frozen source model/threshold on predeclared held-source DEVELOPMENT only."""
import gc
import numpy as np
import torch

from .common import ROOT, read_json, save_json, sha256, now, choose_device,digest
from .image_classification import restore_trainable
from .metrics import classification
from .publication_train import probability_metrics
from .publication_dense_data import prepare_dense_rows
from .publication_dense_train import build_system, native_evaluation, scientific_hashes
from .publication_normal_bank import OrientationNormalBank


def evaluate_development(settings, folder, target):
    allowed = set(settings['permitted_target_development_ids'])
    if not target or any(r['partition'] != 'target_development' or r['id'] not in allowed for r in target):
        raise PermissionError('Unregistered or sealed dense evaluation identities')
    if scientific_hashes() != settings['scientific_hashes']:
        raise RuntimeError('Registered dense implementation changed')
    run = read_json(folder / 'run.json')
    if set(run['training_ids'] + run['validation_ids']) & {r['id'] for r in target}:
        raise PermissionError('Source/held-source identity leakage')
    if sha256(folder / 'best.pt') != run['checkpoint_sha256']:
        raise RuntimeError('Dense checkpoint changed')
    destination = folder / 'target_development.json'
    if destination.exists(): return read_json(destination)
    device = choose_device(settings['device'])
    bank = OrientationNormalBank.from_recipe(folder / 'normal_bank_recipe.json', run['training_ids']) if run['model'].get('reference') else None
    model, identity, _ = build_system(run['model'], run['seed'], settings['input_size'], device, bank=bank)
    if digest(identity) != digest(run['encoder']): raise RuntimeError('Dense encoder identity changed')
    restore_trainable(model, torch.load(folder / 'best.pt', map_location='cpu', weights_only=True))
    examples = prepare_dense_rows(target, settings['input_size'])
    logits, local = native_evaluation(model, examples, device, settings['batch_size'])
    y = np.array([r['target'] for r in target])
    result = {'at': now(), 'checkpoint_sha256': run['checkpoint_sha256'],
              'scope': 'Predeclared held-source DEVELOPMENT; no model/threshold fitting on target',
              'source_logit_threshold': run['source_logit_threshold'],
              'metrics': classification(y, logits, run['source_logit_threshold']), 'localization': local,
              'raw_calibration': probability_metrics(y, logits),
              'source_temperature_calibration': probability_metrics(y, logits, run['source_temperature'])}
    np.savez_compressed(folder / 'target_development_predictions.npz', ids=np.array([r['id'] for r in target]),
                        groups=np.array([r['split_group'] for r in target]), y=y, logits=logits)
    save_json(destination, result)
    del model, bank, examples; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()
    return result
