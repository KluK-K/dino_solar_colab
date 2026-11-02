"""Frozen encoders with explicit weights, full-image transforms and per-image caches."""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import ROOT, choose_device, config, digest, read_json, save_json, sha256
from .data import IGNORE, load_image, load_mask, records

def geometry(width, height, size):
    scale = min(size / width, size / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    return {'width': width, 'height': height, 'w': w, 'h': h,
            'left': (size - w) // 2, 'top': (size - h) // 2, 'size': size}

def image_tensor(image, size):
    gray = image.convert('L')
    g = geometry(*gray.size, size)
    canvas = Image.new('L', (size, size), color=0)
    canvas.paste(gray.resize((g['w'], g['h']), Image.Resampling.BICUBIC), (g['left'], g['top']))
    array = np.array(canvas, dtype=np.float32) / 255
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (tensor - mean) / std, g

def patch_geometry(g, patch_size):
    size = g['size']
    if size % patch_size:
        raise ValueError('Input resolution must be a multiple of patch size')
    valid = torch.zeros(1, 1, size, size)
    valid[:, :, g['top']:g['top'] + g['h'], g['left']:g['left'] + g['w']] = 1
    valid = F.avg_pool2d(valid, patch_size, patch_size).squeeze().numpy() > .999
    grid = size // patch_size
    yy, xx = np.meshgrid(np.arange(grid) * patch_size + patch_size / 2,
                          np.arange(grid) * patch_size + patch_size / 2, indexing='ij')
    coords = np.stack([(xx - g['left']) / g['w'], (yy - g['top']) / g['h']], axis=-1).astype(np.float32)
    return valid, coords

def patch_targets(mask, g, patch_size):
    # Max pooling preserves a one-pixel annotated crack at coarser resolution.
    flags = []
    for value in [1, 2, IGNORE]:
        native = torch.from_numpy((mask == value).astype(np.float32))[None, None]
        resized = F.adaptive_max_pool2d(native, (g['h'], g['w']))
        padded = torch.zeros(1, 1, g['size'], g['size'])
        padded[:, :, g['top']:g['top'] + g['h'], g['left']:g['left'] + g['w']] = resized
        flags.append(F.max_pool2d(padded, patch_size, patch_size).squeeze().numpy() > 0)
    valid, _ = patch_geometry(g, patch_size)
    crack, other, unknown = flags
    output = np.zeros(valid.shape, dtype=np.int64)
    output[other] = 2
    output[unknown | ~valid] = IGNORE
    output[crack & valid] = 1
    return output

def split_tokens(hidden, grid, registers):
    expected = 1 + registers + grid * grid
    if hidden.shape[1] != expected:
        raise ValueError(f'Expected {expected} tokens, got {hidden.shape[1]}; check CLS/register offsets')
    return hidden[:, 0], hidden[:, 1 + registers:].reshape(hidden.shape[0], grid, grid, hidden.shape[-1])

def reproject(score_map, g):
    score = torch.as_tensor(score_map, dtype=torch.float32)[None, None]
    full = F.interpolate(score, (g['size'], g['size']), mode='bilinear', align_corners=False)
    crop = full[:, :, g['top']:g['top'] + g['h'], g['left']:g['left'] + g['w']]
    return F.interpolate(crop, (g['height'], g['width']), mode='bilinear', align_corners=False).squeeze().numpy()

class FrozenEncoder:
    def __init__(self, name='dinov3_small', size=336, device='auto', model_path=None):
        from transformers import AutoModel
        from huggingface_hub import HfApi, hf_hub_download
        self.name, self.size = name, size
        if name in config().get('disabled_encoders', []):
            raise PermissionError(f'{name} is disabled by the user; no weights will be downloaded')
        self.device = choose_device(device)
        torch.set_num_threads(4)
        if name == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = torch.nn.Identity()
            path = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
            self.identity = {'name': name, 'weights': 'IMAGENET1K_V1', 'sha256': sha256(path)}
            self.patch_size = 32
        else:
            spec = config()['models'][name]
            if model_path:
                source = str(Path(model_path).expanduser().resolve())
                checkpoint = Path(source) / 'model.safetensors'
                if not checkpoint.exists():
                    raise ValueError('--model-path must be a Transformers checkpoint folder with config.json and model.safetensors')
                kwargs = {'local_files_only': True}
                revision = 'local:' + sha256(checkpoint)
            else:
                source = spec['id']
                revision = spec['revision'] or HfApi().model_info(source).sha
                kwargs = {'revision': revision}
                checkpoint = Path(hf_hub_download(source, 'model.safetensors', revision=revision))
            # Gated access errors propagate: no random initialization or substitute model.
            self.model = AutoModel.from_pretrained(source, use_safetensors=True, **kwargs)
            self.patch_size = self.model.config.patch_size
            self.identity = {'name': name, 'model_id': source, 'revision': revision,
                             'weights_sha256': sha256(checkpoint), 'config': self.model.config.to_dict()}
        if name != 'resnet18' and size % self.patch_size:
            raise ValueError('Resolution incompatible with patch size')
        self.model.eval().requires_grad_(False).to(self.device)
        self.identity.update(input_size=size, device=str(self.device), dtype='float32',
                             preprocessing='whole-image grayscale->RGB letterbox bicubic ImageNet normalization',
                             code_sha256=sha256(Path(__file__)))
        self.key = digest({k: v for k, v in self.identity.items() if k != 'device'})

    def encode(self, images):
        prepared = [image_tensor(im, self.size) for im in images]
        batch = torch.stack([t for t, _ in prepared]).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            if self.name == 'resnet18':
                x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(batch))))
                x = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
                patch = x.permute(0, 2, 3, 1)
                cls = self.model.avgpool(x).flatten(1)
                # ResNet uses ceil spatial strides, so 336 gives 11x11 rather than exact 32 grid.
                grid = patch.shape[1]
            else:
                output = self.model(pixel_values=batch)
                grid = self.size // self.patch_size
                registers = getattr(self.model.config, 'num_register_tokens', 0)
                cls, patch = split_tokens(output.last_hidden_state, grid, registers)
        cls, patch = cls.cpu().float().numpy(), patch.cpu().float().numpy()
        if not np.isfinite(cls).all() or not np.isfinite(patch).all():
            raise ValueError('Encoder returned nonfinite features')
        seconds = time.perf_counter() - started
        results = []
        for i, (_, g) in enumerate(prepared):
            if self.name == 'resnet18':
                # Encoder feature-map validity follows exact letterbox fractions.
                valid_pixels = np.zeros((self.size, self.size), np.float32)
                valid_pixels[g['top']:g['top']+g['h'], g['left']:g['left']+g['w']] = 1
                valid = F.adaptive_avg_pool2d(torch.from_numpy(valid_pixels)[None, None], (grid, grid)).squeeze().numpy() > .999
                yy, xx = np.meshgrid((np.arange(grid) + .5) * self.size / grid, (np.arange(grid) + .5) * self.size / grid, indexing='ij')
                coords = np.stack([(xx-g['left'])/g['w'], (yy-g['top'])/g['h']], -1).astype(np.float32)
            else:
                valid, coords = patch_geometry(g, self.patch_size)
            if not valid.any():
                raise ValueError('No fully valid patch remains after letterboxing')
            results.append({'cls': cls[i], 'patch': patch[i].astype(np.float16),
                            'mean': patch[i][valid].mean(0), 'valid': valid, 'coords': coords,
                            'geometry': g, 'seconds': seconds / len(images)})
        return results

def cache_path(encoder, row):
    return ROOT / 'cache' / encoder.name / encoder.key / (digest({'id': row['id'], 'image': row['image_sha256']}) + '.npz')

def extract(dataset, encoder, partition, batch_size=8, *, evaluation=False):
    rows = records(dataset, partition, evaluation=evaluation)
    pending = [r for r in rows if not cache_path(encoder, r).exists()]
    for start in range(0, len(pending), batch_size):
        selected = pending[start:start + batch_size]
        encoded = encoder.encode([load_image(row) for row in selected])
        for row, value in zip(selected, encoded):
            path = cache_path(encoder, row); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, cls=value['cls'], patch=value['patch'], mean=value['mean'],
                                valid=value['valid'], coords=value['coords'], geometry=json.dumps(value['geometry']),
                                seconds=value['seconds'])
        print(f'{dataset}/{partition}/{encoder.name}: {min(start + batch_size, len(pending))}/{len(pending)} new images', flush=True)
    save_json(ROOT / 'cache' / encoder.name / encoder.key / 'encoder.json', encoder.identity)
    return rows

def read_feature(encoder, row):
    with np.load(cache_path(encoder, row), allow_pickle=False) as f:
        values = {k: f[k].copy() for k in f.files}
    values['geometry'] = json.loads(str(values['geometry']))
    values['patch'] = values['patch'].astype(np.float32)
    return values

def vectors(encoder, rows, representation='concat'):
    result = []
    for row in rows:
        feature = read_feature(encoder, row)
        if representation == 'cls':
            result.append(feature['cls'])
        elif representation == 'mean':
            result.append(feature['mean'])
        elif representation == 'concat':
            result.append(np.concatenate([feature['cls'], feature['mean']]))
        else:
            raise ValueError(representation)
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite embedding')
    return result
