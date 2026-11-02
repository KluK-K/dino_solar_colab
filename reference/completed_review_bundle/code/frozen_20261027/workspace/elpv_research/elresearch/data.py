"""Canonical labels, immutable acquisition splits and a sealed evaluation loader."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .common import ROOT, config, digest, read_json, save_json, sha256

IGNORE = 255

def canonical_mask(array, dataset):
    if array.ndim == 3:
        if not np.array_equal(array[..., 0], array[..., 1]) or not np.array_equal(array[..., 0], array[..., 2]):
            raise ValueError('RGB annotation requires an explicit documented color mapping')
        array = array[..., 0]
    values = set(np.unique(array).tolist())
    if dataset == 'ucf':
        if not values <= {0, 1, 2, 3, 4}:
            raise ValueError(f'Unexpected UCF mask indices: {sorted(values)}')
        output = np.where(array == 1, 1, np.where(array == 0, 0, 2)).astype(np.uint8)
        ambiguous = False
    elif dataset == 'scdd':
        if not values <= set(range(29)):
            raise ValueError(f'Unexpected SCDD mask indices: {sorted(values)}')
        output = np.full(array.shape, IGNORE, dtype=np.uint8)
        normal = {0, 1, 2, 3, 4, 9, 24}
        crack = {10, 14}
        other = {11, 12, 13, 15, 16, 17, 18, 19, 20, 25, 26, 27}
        output[np.isin(array, list(normal))] = 0
        output[np.isin(array, list(crack))] = 1
        output[np.isin(array, list(other))] = 2
        # 'star' has not been verified as a semantic alias of crack.
        ambiguous = bool((array == 28).any())
    else:
        raise ValueError(dataset)
    return output, ambiguous

def _connected_groups(records):
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]
    def join(a, b):
        parent[find(a)] = find(b)
    seen = {}
    for row in records:
        group = row['group_id']; find(group)
        h = row['pixel_sha256']
        if h in seen:
            join(group, seen[h])
        seen[h] = group
    for row in records:
        row['split_group'] = find(row['group_id'])

def make_splits(records, seed):
    labels = np.array([r['target'] for r in records])
    groups = np.array([r['split_group'] for r in records])
    if len(set(groups)) < 5 or len(set(labels)) < 2:
        raise ValueError('Need at least five acquisition groups and both labels to create a 60/20/20 split')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(labels)), labels, groups))
    val, test = set(folds[0][1].tolist()), set(folds[1][1].tolist())
    for i, row in enumerate(records):
        row['partition'] = 'validation' if i in val else 'test' if i in test else 'train'
    split_groups = {part: {r['split_group'] for r in records if r['partition'] == part} for part in ['train', 'validation', 'test']}
    assert not (split_groups['train'] & split_groups['validation'] or split_groups['train'] & split_groups['test'] or split_groups['validation'] & split_groups['test'])
    for part in split_groups:
        if len({r['target'] for r in records if r['partition'] == part}) < 2:
            raise ValueError(f'{part} has only one class; expand acquisition instead of repeatedly choosing favorable seeds')
    return records

def audit(dataset):
    imported = read_json(ROOT / 'data' / dataset / 'downloaded.json')
    records, failures = [], []
    for row in imported['pairs']:
        try:
            image_path, mask_path = ROOT / row['image_path'], ROOT / row['mask_path']
            with Image.open(image_path) as im:
                gray = np.asarray(im.convert('L')).copy()
            with Image.open(mask_path) as im:
                raw_mask = np.asarray(im).copy()
            mask, ambiguous = canonical_mask(raw_mask, dataset)
            if gray.shape != mask.shape:
                raise ValueError(f'Image/mask shape mismatch: {gray.shape}/{mask.shape}')
            crack = bool((mask == 1).any())
            if ambiguous and not crack:
                raise ValueError('Unknown potential crack class; image excluded from binary task')
            records.append({**row, 'dataset': dataset, 'target': int(crack),
                            'image_sha256': sha256(image_path), 'mask_sha256': sha256(mask_path),
                            'pixel_sha256': hashlib.sha256(gray.tobytes()).hexdigest(),
                            'shape': list(gray.shape), 'crack_area_fraction': float((mask == 1).mean()),
                            'annotation_type': 'indexed_pixel_mask', 'module_id': None})
        except Exception as e:
            failures.append({'id': row['id'], 'reason': str(e)})
    _connected_groups(records)
    seen = set()
    unique = []
    for row in records:
        if row['pixel_sha256'] not in seen:
            seen.add(row['pixel_sha256']); unique.append(row)
    records = sorted(unique, key=lambda r: r['id'])
    signature = digest({'source': imported['commit'], 'rows': [(r['id'], r['image_sha256'], r['mask_sha256'], r['split_group']) for r in records],
                        'seed': config()['seed'], 'label_mapping': sha256(Path(__file__))})
    path = ROOT / 'data' / dataset / 'manifest.json'
    if path.exists():
        old = read_json(path)
        if old['signature'] != signature:
            raise RuntimeError('Existing manifest is immutable; use a separate cohort for expanded or changed data')
        return old
    records = make_splits(records, config()['seed'])
    manifest = {'dataset': dataset, 'signature': signature, 'records': records, 'excluded': failures,
                'source_commit': imported['commit'], 'evaluation_state': 'sealed',
                'group_semantics': 'conservative acquisition family; physical module identity unconfirmed',
                'publication_eligible': bool(imported.get('publication_eligible'))}
    save_json(path, manifest)
    summary = {'dataset': dataset, 'original_pairs': len(imported['pairs']), 'usable_unique': len(records),
               'excluded': failures, 'publication_eligible': manifest['publication_eligible'],
               'group_semantics': manifest['group_semantics'],
               'partitions': {p: {'images': sum(r['partition'] == p for r in records),
                                  'positive': sum(r['partition'] == p and r['target'] == 1 for r in records),
                                  'groups': len({r['split_group'] for r in records if r['partition'] == p})}
                              for p in ['train', 'validation', 'test']}}
    save_json(ROOT / 'artifacts' / f'{dataset}_audit.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return manifest

def import_elpv():
    import pandas as pd
    old = ROOT.parent / 'elpv_midterm'
    split = read_json(old / 'data/split_manifest.json')
    metadata = pd.read_csv(old / 'data/metadata.csv').set_index('path')
    records = []
    for name in split['development_ids']:
        row = metadata.loc[name]
        records.append({'id': 'elpv/' + name, 'dataset': 'elpv', 'image_path': str(old / 'data/source' / name),
                        'target': int(row['target']), 'partition': 'development', 'split_group': row['pixel_sha256'],
                        'stratum': row['stratum'], 'cell_type': row['cell_type'], 'expert_label': float(row['expert_label']),
                        'pixel_sha256': row['pixel_sha256'], 'image_sha256': sha256(old / 'data/source' / name),
                        'mask_path': None, 'real_or_synthetic': 'real', 'annotation_type': 'expert_review_label'})
    result = {'dataset': 'elpv', 'signature': digest({'old_split': sha256(old / 'data/split_manifest.json'), 'records': records}),
              'records': records, 'test_rows': len(split['test_ids']), 'evaluation_state': 'course_test_sealed',
              'task': 'requires_review_not_crack', 'publication_eligible': True}
    save_json(ROOT / 'data/elpv/manifest.json', result)
    return result

def records(dataset, partition='train', *, evaluation=False):
    manifest = read_json(ROOT / 'data' / dataset / 'manifest.json')
    if partition in {'test', 'external'} and not evaluation:
        raise PermissionError('Evaluation is sealed; use the frozen evaluation command')
    if dataset == 'elpv' and partition != 'development':
        raise PermissionError('ELPV course test cannot be opened by research tools')
    if partition == 'external':
        return manifest['records']
    result = [r for r in manifest['records'] if r['partition'] == partition]
    if not result:
        raise ValueError(f'Empty partition: {dataset}/{partition}')
    return result

def load_image(row):
    path = ROOT / row['image_path']
    if sha256(path) != row['image_sha256']:
        raise ValueError('Image modified after audit')
    return Image.open(path).convert('L')

def load_mask(row):
    if not row.get('mask_path'):
        raise ValueError('This task has no localization labels')
    path = ROOT / row['mask_path']
    if sha256(path) != row['mask_sha256']:
        raise ValueError('Mask modified after audit')
    with Image.open(path) as im:
        return canonical_mask(np.asarray(im), row['dataset'])[0]
