"""Synchronized whole-field D4 transforms and deterministic validation pixels."""
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .common import ROOT, sha256
from .data import IGNORE, load_image, load_mask
from .encoders import image_tensor
from .publication_models import d4_view


def mask_target(mask, geometry):
    g = geometry
    target = F.interpolate(torch.as_tensor(mask.astype(np.float32))[None, None],
                           (g['h'], g['w']), mode='nearest')[0, 0].long()
    cracks = F.adaptive_max_pool2d(torch.as_tensor((mask == 1).astype(np.float32))[None, None],
                                 (g['h'], g['w']))[0, 0] > 0
    target[cracks] = 1  # Preserve thin annotated cracks; shared by all systems.
    canvas = torch.full((g['size'], g['size']), IGNORE, dtype=torch.long)
    canvas[g['top']:g['top'] + g['h'], g['left']:g['left'] + g['w']] = target
    return canvas


def rotate_geometry(g, code):
    if not 0 <= int(code) < 8:
        raise ValueError('D4 code must be 0..7')
    result = dict(g)
    for _ in range(int(code) % 4):
        old = dict(result)
        result.update(left=old['top'], top=old['size'] - old['left'] - old['w'],
                      w=old['h'], h=old['w'], width=old['height'], height=old['width'])
    if code >= 4:
        result['left'] = result['size'] - result['left'] - result['w']
    return result


def prepare_dense_rows(rows, size):
    result = []
    for row in rows:
        if row['partition'] not in {'train', 'validation', 'target_development'}:
            raise PermissionError('Dense development cannot open confirmation images')
        if row['dataset'] not in {'ucf', 'scdd'}:
            raise PermissionError('Only registered crack-mask source domains are allowed')
        image, g = image_tensor(load_image(row), size)
        mask = load_mask(row)
        target = mask_target(mask, g)
        valid = torch.zeros(size, size, dtype=torch.bool)
        valid[g['top']:g['top'] + g['h'], g['left']:g['left'] + g['w']] = True
        # At most 8192 known pixel positions per image, selected uniformly without
        # reading class values beyond the known/ignored eligibility mask. Reused
        # across all rounds/models. Selection metric is explicitly approximate.
        candidates = np.flatnonzero((target != IGNORE).numpy().ravel())
        if not len(candidates):
            raise ValueError('No known source annotation pixels')
        import hashlib
        seed = int(hashlib.sha256(row['id'].encode()).hexdigest()[:16], 16)
        selection = np.sort(np.random.default_rng(seed).choice(candidates, min(8192, len(candidates)), replace=False))
        structure = None
        if row['dataset'] == 'scdd':
            with Image.open(ROOT / row['mask_path']) as im:
                structure = np.isin(np.asarray(im), [4, 9])
        result.append({'row': row, 'image': image, 'geometry': g, 'target': target,
                       'valid': valid, 'native_mask': mask, 'selection_indices': selection})
        result[-1]['native_structure'] = structure
    return result


def augmented_batch(items, codes, device):
    return {'images': torch.stack([d4_view(x['image'], int(v)) for x, v in zip(items, codes)]).to(device),
            'target': torch.stack([d4_view(x['target'], int(v)) for x, v in zip(items, codes)]).to(device),
            'valid': torch.stack([d4_view(x['valid'], int(v)) for x, v in zip(items, codes)]).to(device),
            'geometries': [rotate_geometry(x['geometry'], int(v)) for x, v in zip(items, codes)],
            'groups': [(x['row']['split_group'], int(v)) for x, v in zip(items, codes)],
            'y': torch.tensor([x['row']['target'] for x in items], dtype=torch.float32, device=device)}
