"""Matched-supervision heads and source-only normal references."""
from pathlib import Path
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, choose_device, config, digest, read_json, save_json, sha256, now
from .data import IGNORE, records, load_mask
from .encoders import read_feature, patch_targets, extract
from .metrics import classification, recall_threshold

def budget_rows(rows, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('Budget must lie in (0,1]')
    if fraction == 1:
        return list(rows)
    rng = np.random.default_rng(seed); chosen = []
    for label in [0, 1]:
        group = [r for r in rows if r['target'] == label]
        n = max(1, int(round(len(group) * fraction)))
        chosen.extend(group[i] for i in rng.permutation(len(group))[:n])
    return sorted(chosen, key=lambda r: r['id'])

def normalized(x):
    return F.normalize(torch.as_tensor(x, dtype=torch.float32), dim=-1)

class NormalBank:
    def __init__(self, items, encoder_key):
        self.items = items; self.encoder_key = encoder_key
        if not items:
            raise ValueError('No verified normal patches available in source training data')
        self.cls = torch.stack([item['cls'] for item in items])

    @classmethod
    def build(cls, rows, encoder, limit=256):
        items = []
        for row in rows:
            if row['partition'] != 'train':
                raise PermissionError('Normal reference bank is restricted to source training images')
            feature = read_feature(encoder, row)
            target = patch_targets(load_mask(row), feature['geometry'], encoder.patch_size)
            normal = (target == 0) & feature['valid']
            if normal.sum() < 10:
                continue
            items.append({'id': row['id'], 'origin_id': row['origin_id'], 'group': row['split_group'],
                          'cls': normalized(feature['cls']), 'patch': normalized(feature['patch'][normal]),
                          'coords': torch.from_numpy(feature['coords'][normal]).float()})
        # Label-independent farthest-first coverage, among normal-patch-eligible frames.
        if len(items) > limit:
            vec = torch.stack([i['cls'] for i in items]); selected = [0]
            distance = 1 - vec @ vec[0]
            for _ in range(limit - 1):
                idx = int(distance.argmax()); selected.append(idx)
                distance = torch.minimum(distance, 1 - vec @ vec[idx]); distance[selected] = -1
            items = [items[i] for i in selected]
        return cls(items, encoder.key)

    def residual(self, feature, query_group=None, topk=5, geometry_weight=.1, device='cpu'):
        eligible = [i for i, item in enumerate(self.items) if item['group'] != query_group]
        if not eligible:
            raise ValueError('No references outside the query acquisition group; self-reference is prohibited')
        similarities = self.cls[eligible] @ normalized(feature['cls'])
        selected = [eligible[i] for i in similarities.topk(min(topk, len(eligible))).indices.tolist()]
        bank = torch.cat([self.items[i]['patch'] for i in selected]).to(device)
        coords = torch.cat([self.items[i]['coords'] for i in selected]).to(device)
        query = normalized(feature['patch'].reshape(-1, feature['patch'].shape[-1])).to(device)
        xy = torch.from_numpy(feature['coords'].reshape(-1, 2)).to(device)
        appearance = 1 - query @ bank.T
        positional = ((xy[:, None] - coords[None]) ** 2).sum(-1)
        match = (appearance + geometry_weight * positional).argmin(1)
        residual = query - bank[match]
        distances = appearance[torch.arange(len(query), device=device), match, None]
        return torch.cat([query, residual, xy, distances], -1).cpu().reshape(*feature['patch'].shape[:2], -1)

    def save(self, path):
        torch.save({'items': self.items, 'encoder_key': self.encoder_key}, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        return cls(state['items'], state['encoder_key'])

class InspectionHead(nn.Module):
    def __init__(self, dimensions, hidden=128, top_fraction=.05):
        super().__init__()
        self.top_fraction = top_fraction
        self.local = nn.Sequential(nn.Linear(2 * dimensions + 3, hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, 3))
        self.global_head = nn.Linear(2 * dimensions + 1, 1)

    def forward(self, inputs, cls, mean, valid):
        logits = self.local(inputs)
        crack = logits.softmax(-1)[..., 1]
        summary = []
        for values, keep in zip(crack, valid):
            selected = values[keep]
            if not selected.numel():
                raise ValueError('No valid local token')
            k = max(1, math.ceil(len(selected) * self.top_fraction))
            summary.append(selected.topk(k).values.mean())
        aggregate = torch.stack(summary)[:, None]
        image_logits = self.global_head(torch.cat([cls, mean, aggregate], -1)).squeeze(-1)
        return image_logits, logits

def prepare_rows(rows, encoder, bank, method, geometry_weight=.1, device='cpu'):
    result = []
    for row in rows:
        f = read_feature(encoder, row); patch = normalized(f['patch'])
        if method == 'reference':
            local = bank.residual(f, query_group=row['split_group'], geometry_weight=geometry_weight, device=device)
        elif method == 'local':
            local = torch.cat([patch, torch.zeros_like(patch), torch.from_numpy(f['coords']), torch.zeros(*patch.shape[:2], 1)], -1)
        else:
            raise ValueError(method)
        target = patch_targets(load_mask(row), f['geometry'], encoder.patch_size)
        result.append({'row': row, 'inputs': local.reshape(-1, local.shape[-1]),
                       'cls': torch.from_numpy(f['cls']).float(), 'mean': torch.from_numpy(f['mean']).float(),
                       'valid': torch.from_numpy(f['valid'].reshape(-1)),
                       'target': torch.from_numpy(target.reshape(-1)), 'geometry': f['geometry']})
    return result

def batch(items, device):
    keys = ['inputs', 'cls', 'mean', 'valid', 'target']
    return {k: torch.stack([item[k] for item in items]).to(device) for k in keys}

def head_predictions(head, items, device, batch_size=8):
    scores, maps = [], []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            selected = items[start:start + batch_size]; b = batch(selected, device)
            image, local = head(b['inputs'], b['cls'], b['mean'], b['valid'])
            scores.extend(image.sigmoid().cpu().numpy().tolist())
            maps.extend(local.softmax(-1)[..., 1].cpu().numpy())
    return np.array(scores), maps

def train_head(dataset, encoder, method='reference', budget=1., seed=11, epochs=None, geometry_weight=None):
    if dataset == 'elpv':
        raise ValueError('ELPV review labels cannot supervise a crack localization head')
    cfg = config(); epochs = epochs or cfg['epochs']
    geometry_weight = cfg['geometry_weight'] if geometry_weight is None else geometry_weight
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = encoder.device
    train = budget_rows(records(dataset, 'train'), budget, seed); validation = records(dataset, 'validation')
    extract(dataset, encoder, 'train', cfg['batch_size']); extract(dataset, encoder, 'validation', cfg['batch_size'])
    manifest = read_json(ROOT / 'data' / dataset / 'manifest.json')
    signature = {'encoder': encoder.identity, 'source_manifest': manifest['signature'], 'method': method,
                 'budget': budget, 'seed': seed, 'epochs': epochs, 'geometry_weight': geometry_weight,
                 'config': cfg, 'code': sha256(Path(__file__))}
    run_id = digest(signature)[:20]; folder = ROOT / 'artifacts/runs' / run_id
    if (folder / 'frozen.json').exists():
        raise RuntimeError('Frozen run cannot be trained again')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bank = NormalBank.build(train, encoder, cfg['reference_images']) if method == 'reference' else None
    if bank:
        bank.save(folder / 'normal_bank.pt')
    train_items = prepare_rows(train, encoder, bank, method, geometry_weight, device)
    validation_items = prepare_rows(validation, encoder, bank, method, geometry_weight, device)
    dimensions = train_items[0]['cls'].numel()
    head = InspectionHead(dimensions, cfg['hidden_dim'], cfg['top_fraction']).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    counts = sum((torch.bincount(item['target'][item['target'] != IGNORE], minlength=3) for item in train_items), torch.zeros(3, dtype=torch.int64))
    weight = counts.clamp_min(1).float().rsqrt(); weight /= weight.mean(); weight = weight.to(device)
    history, best, patience = [], -np.inf, 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        head.train(); losses = []
        permutation = rng.permutation(len(train_items))
        for start in range(0, len(permutation), cfg['batch_size']):
            selected = [train_items[i] for i in permutation[start:start + cfg['batch_size']]]
            b = batch(selected, device)
            image, local = head(b['inputs'], b['cls'], b['mean'], b['valid'])
            y = torch.tensor([item['row']['target'] for item in selected], device=device, dtype=torch.float32)
            patch_loss = (F.cross_entropy(local.flatten(0, 1), b['target'].flatten(), weight=weight, ignore_index=IGNORE)
                          if (b['target'] != IGNORE).any() else local.sum() * 0)
            loss = F.binary_cross_entropy_with_logits(image, y) + patch_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        scores, _ = head_predictions(head, validation_items, device, cfg['batch_size'])
        metric = classification([r['target'] for r in validation], scores, .5)
        auc = metric['auc']
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'validation_auc': auc})
        print(f'{method} epoch {epoch + 1}: loss={np.mean(losses):.4f}, validation AUC={auc:.4f}', flush=True)
        save_json(folder / 'history.json', history)
        if auc > best:
            best = auc; patience = 0
            torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, folder / 'head.pt')
        else:
            patience += 1
            if patience >= cfg['patience']:
                break
    head.load_state_dict(torch.load(folder / 'head.pt', map_location=device, weights_only=True))
    scores, _ = head_predictions(head, validation_items, device, cfg['batch_size'])
    y = np.array([r['target'] for r in validation]); threshold = recall_threshold(y, scores, cfg['review_recall'])
    result = {'run_id': run_id, 'kind': 'head', 'dataset': dataset, 'task': 'crack_presence', 'method': method,
              **signature, 'encoder_key': encoder.key, 'dimensions': dimensions,
              'training_ids': [r['id'] for r in train], 'calibration_ids': [r['id'] for r in validation],
              'threshold': threshold, 'validation_metrics': classification(y, scores, threshold),
              'checkpoint_sha256': sha256(folder / 'head.pt'),
              'normal_bank_sha256': sha256(folder / 'normal_bank.pt') if bank else None,
              'seconds': time.perf_counter() - started, 'created_at': now(),
              'publication_eligible': manifest.get('publication_eligible', False),
              'supervision': 'image labels and three-class pixel-mask supervision; local baseline has identical supervision'}
    save_json(folder / 'run.json', result)
    np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array([r['id'] for r in validation]), y=y, scores=scores)
    return result
