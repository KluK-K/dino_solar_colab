"""Frozen, label-independent representation controls with pinned official weights."""
from pathlib import Path
import importlib.metadata

import numpy as np
import torch
from torch.nn import functional as F

from .common import ROOT, digest, read_json, save_json, sha256
from .classical_features import FeatureBank
from .data import load_image
from .encoders import image_tensor, split_tokens
from .publication_models import d4_view


def geometric_patch_weights(geometry, patch_size):
    size = geometry['size']
    if size % patch_size: raise ValueError('Resolution must be divisible by patch size')
    valid = torch.zeros(1, 1, size, size)
    valid[:, :, geometry['top']:geometry['top']+geometry['h'],
          geometry['left']:geometry['left']+geometry['w']] = 1
    return F.avg_pool2d(valid, patch_size, patch_size)[0, 0]


def pool_dino_tokens(hidden, grid, registers, weights):
    """Exclude CLS/registers from an area-weighted mean of image-overlapping patches."""
    cls, patches = split_tokens(hidden, grid, registers)
    if weights.shape != patches.shape[:3]: raise ValueError('Patch weights do not match token geometry')
    if not torch.isfinite(weights).all() or (weights < 0).any(): raise ValueError('Invalid patch weights')
    total = weights.sum((1, 2))
    if (total <= 0).any(): raise ValueError('Image has no valid patch area')
    mean = (patches * weights[..., None]).sum((1, 2)) / total[:, None]
    return cls, mean


def normalized_representation(cls, mean, pooling):
    cls, mean = np.asarray(cls, np.float32), np.asarray(mean, np.float32)
    if pooling in {'cls', 'global'}: value = cls
    elif pooling == 'patch_mean': value = mean
    elif pooling == 'concat': value = np.concatenate([cls, mean], axis=-1)
    else: raise ValueError('Unknown pooling: '+pooling)
    # Per-image, non-affine normalization; never estimate target-domain statistics.
    result = F.layer_norm(torch.from_numpy(value), (value.shape[-1],), eps=1e-5).numpy()
    if not np.isfinite(result).all(): raise ValueError('Nonfinite embedding')
    return result


def official_identities():
    base = ROOT/'artifacts/publication_20261017'
    small = read_json(base/'calibration_v1/runs/dinov3_frozen_lr1_seed11/run.json')['encoder']
    large = read_json(base/'official_dinov3_base_identity.json')
    cnn = read_json(base/'calibration_v1/runs/resnet18_full_lr1_seed11/run.json')['encoder']
    return {
        'dinov3_small': {'name':'dinov3_small', 'model_id':small['model_id'], 'revision':small['revision'],
            'weights_sha256':small['weights_sha256'], 'dimensions':384, 'patch_size':16, 'registers':4},
        'dinov3_base': {'name':'dinov3_base', 'model_id':large['model_id'], 'revision':large['revision'],
            'weights_sha256':large['weights_sha256'], 'config_sha256':large['config_sha256'],
            'dimensions':768, 'patch_size':16, 'registers':4},
        'resnet18': {'name':'resnet18', 'weights':'ResNet18_Weights.IMAGENET1K_V1',
            'weights_sha256':cnn['weights_sha256'], 'dimensions':512},
    }


class FrozenRepresentationEncoder:
    def __init__(self, identity, size, device='cpu'):
        self.identity, self.size, self.device = identity, int(size), torch.device(device)
        name = identity['name']
        if name not in {'dinov3_small','dinov3_base','resnet18'}: raise PermissionError('Unregistered encoder')
        if name == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            path = Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
            if not path.exists() or sha256(path) != identity['weights_sha256']:
                raise RuntimeError('Pinned ImageNet checkpoint is missing or changed')
            self.model = resnet18(weights=None)
            self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            self.model.fc = torch.nn.Identity()
        else:
            from transformers import AutoModel
            from huggingface_hub import hf_hub_download
            if self.size % identity['patch_size']: raise ValueError('Invalid ViT resolution')
            path = Path(hf_hub_download(identity['model_id'], 'model.safetensors',
                        revision=identity['revision'], local_files_only=True))
            if sha256(path) != identity['weights_sha256']: raise RuntimeError('Pinned DINOv3 weights changed')
            if identity.get('config_sha256'):
                cfg = Path(hf_hub_download(identity['model_id'], 'config.json',
                    revision=identity['revision'], local_files_only=True))
                if sha256(cfg) != identity['config_sha256']: raise RuntimeError('Pinned DINOv3 configuration changed')
            self.model = AutoModel.from_pretrained(identity['model_id'], revision=identity['revision'],
                local_files_only=True, use_safetensors=True)
            cfg = self.model.config
            if (cfg.hidden_size, cfg.patch_size, cfg.num_register_tokens) != (
                    identity['dimensions'], identity['patch_size'], identity['registers']):
                raise RuntimeError('Unexpected DINOv3 architecture')
        self.model.eval().requires_grad_(False).to(self.device)

    def encode(self, images, views=range(8)):
        prepared = [image_tensor(image, self.size) for image in images]
        batch = torch.stack([p[0] for p in prepared]).to(self.device)
        weights = None
        if self.identity['name'] != 'resnet18':
            weights = torch.stack([geometric_patch_weights(p[1], self.identity['patch_size']) for p in prepared]).to(self.device)
        cls_values, mean_values = [], []
        with torch.inference_mode():
            for view in views:
                data = d4_view(batch, view)
                if weights is None:
                    cls = self.model(data); mean = cls
                else:
                    hidden = self.model(pixel_values=data).last_hidden_state
                    transformed = d4_view(weights[:, None], view)[:, 0]
                    cls, mean = pool_dino_tokens(hidden, self.size//self.identity['patch_size'],
                        self.identity['registers'], transformed)
                cls_values.append(cls.float().cpu().numpy()); mean_values.append(mean.float().cpu().numpy())
        return np.stack(cls_values, axis=1), np.stack(mean_values, axis=1)


def feature_identity(encoder_identity, size):
    return {'kind':'frozen_representation_v1', 'encoder':encoder_identity, 'input_size':size,
        'normalization':'per-image non-affine LayerNorm eps=1e-5 after pooling/concatenation; no fitted scaler',
        'patch_pooling':'image-area-weighted patch mean, including fractional boundary patches; excludes CLS/registers and zero-area padding patches',
        'augmentation':'D4 rot90 k0..3 then optional horizontal flip; transform geometric patch weights identically',
        'feature_code_sha256':sha256(__file__), 'preprocessing_sha256':sha256(ROOT/'elresearch/encoders.py'),
        'd4_code_sha256':sha256(ROOT/'elresearch/publication_models.py'),
        'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','numpy','Pillow']}}


def validate_cached_pair(cls, mean, dimensions):
    if cls.shape != (8,dimensions) or mean.shape != (8,dimensions): raise RuntimeError('Feature cache shape changed')
    if not np.isfinite(cls).all() or not np.isfinite(mean).all(): raise RuntimeError('Nonfinite cached features')


def cached_banks(rows, encoder_identity, size, poolings, device='mps', batch_size=4):
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Unique nonempty source rows required')
    if any(r['partition'] not in {'train','validation','target_development'} for r in rows):
        raise PermissionError('Development extractor cannot open confirmation images')
    identity = feature_identity(encoder_identity, size)
    root = ROOT/'cache/publication_representation_d4'; root.mkdir(parents=True, exist_ok=True)
    paths = [root/(digest({'identity':identity,'image_id':r['id'],'image_sha256':r['image_sha256']})+'.npz') for r in rows]
    missing = [i for i,p in enumerate(paths) if not p.exists() or not p.with_suffix('.json').exists()]
    if missing:
        encoder = FrozenRepresentationEncoder(encoder_identity, size, device)
        for start in range(0,len(missing),batch_size):
            indices = missing[start:start+batch_size]
            cls, mean = encoder.encode([load_image(rows[i]) for i in indices])
            for local,index in enumerate(indices):
                validate_cached_pair(cls[local], mean[local], encoder_identity['dimensions'])
                path = paths[index]; temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, cls=cls[local], mean=mean[local]); temporary.replace(path)
                save_json(path.with_suffix('.json'), {'identity':identity,'id':rows[index]['id'],
                    'image_sha256':rows[index]['image_sha256'],'payload_sha256':sha256(path)})
            print('FROZEN FEATURES',encoder_identity['name'],size,min(start+batch_size,len(missing)),'/',len(missing),flush=True)
        del encoder
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    raw_cls, raw_mean = [], []
    for row,path in zip(rows,paths):
        metadata = read_json(path.with_suffix('.json'))
        if (metadata['identity'] != identity or metadata['id'] != row['id']
                or metadata['image_sha256'] != row['image_sha256'] or metadata['payload_sha256'] != sha256(path)):
            raise RuntimeError('Feature cache identity or payload changed')
        with np.load(path,allow_pickle=False) as stored:
            cls, mean = stored['cls'].copy(), stored['mean'].copy()
        validate_cached_pair(cls,mean,encoder_identity['dimensions']);raw_cls.append(cls);raw_mean.append(mean)
    raw_cls, raw_mean = np.stack(raw_cls), np.stack(raw_mean)
    return {pool:FeatureBank(rows, normalized_representation(raw_cls,raw_mean,pool), {**identity,'pooling':pool}) for pool in poolings}
