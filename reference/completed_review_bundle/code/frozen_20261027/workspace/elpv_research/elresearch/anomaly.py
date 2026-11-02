"""Patch nearest-neighbor baseline, adapted here; not an author-code reproduction."""
import math
import numpy as np
import torch

from .common import ROOT, config, digest, read_json, save_json, sha256, now
from .data import records
from .encoders import extract, read_feature
from .heads import NormalBank, budget_rows
from .metrics import classification, recall_threshold

def anomaly_score(feature, bank, query_group=None):
    eligible = [item for item in bank.items if item['group'] != query_group]
    if not eligible:
        raise ValueError('No normal reference outside the query group')
    memory = torch.cat([item['patch'] for item in eligible])
    query = torch.nn.functional.normalize(torch.from_numpy(feature['patch']).reshape(-1, memory.shape[-1]), dim=-1)
    minimum = torch.full((len(query),), float('inf'))
    # Bounded CPU memory; each chunk calculates visual cosine distance only.
    for start in range(0, len(memory), 2048):
        distance = 1 - query @ memory[start:start + 2048].T
        minimum = torch.minimum(minimum, distance.min(1).values)
    values = minimum.numpy().reshape(feature['valid'].shape)
    valid = values[feature['valid']]
    k = max(1, math.ceil(len(valid) * config()['top_fraction']))
    return float(np.sort(valid)[-k:].mean()), values

def train_anomaly(dataset, encoder, budget=1., seed=11):
    if dataset == 'elpv':
        raise ValueError('Review labels do not provide verified normal patch annotations')
    train = budget_rows(records(dataset, 'train'), budget, seed); validation = records(dataset, 'validation')
    extract(dataset, encoder, 'train', config()['batch_size']); extract(dataset, encoder, 'validation', config()['batch_size'])
    bank = NormalBank.build(train, encoder, config()['reference_images'])
    scores = np.array([anomaly_score(read_feature(encoder, row), bank, row['split_group'])[0] for row in validation])
    y = np.array([r['target'] for r in validation]); threshold = recall_threshold(y, scores, config()['review_recall'])
    manifest = read_json(ROOT / 'data' / dataset / 'manifest.json')
    run_id = digest({'method': 'normal_patch_cosine', 'source': manifest['signature'], 'encoder': encoder.identity,
                     'budget': budget, 'seed': seed, 'code': sha256(__file__)})[:20]
    folder = ROOT / 'artifacts/runs' / run_id; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'frozen.json').exists():
        raise RuntimeError('Frozen run cannot be changed')
    bank.save(folder / 'normal_bank.pt')
    result = {'run_id': run_id, 'kind': 'anomaly', 'dataset': dataset, 'method': 'normal_patch_cosine',
              'encoder': encoder.identity, 'encoder_key': encoder.key, 'budget': budget, 'seed': seed,
              'training_ids': [r['id'] for r in train], 'calibration_ids': [r['id'] for r in validation],
              'source_manifest': manifest['signature'], 'threshold': threshold,
              'validation_metrics': classification(y, scores, threshold),
              'checkpoint_sha256': sha256(folder / 'normal_bank.pt'), 'normal_bank_sha256': sha256(folder / 'normal_bank.pt'),
              'publication_eligible': manifest.get('publication_eligible', False), 'config': config(), 'created_at': now(),
              'supervision': 'normal patches identified using source training pixel annotations',
              'limitation': 'This normal-vs-abnormal baseline also responds to non-crack defects.'}
    save_json(folder / 'run.json', result)
    np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array([r['id'] for r in validation]), y=y, scores=scores)
    return result
