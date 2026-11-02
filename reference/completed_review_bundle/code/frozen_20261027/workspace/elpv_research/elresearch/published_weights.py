"""Strict ImageNet backbone loading; never substitute EL-task-trained weights."""
from pathlib import Path
import torch

from .common import ROOT,read_json,sha256


def backbone_identity():
    info=read_json(ROOT/'models/published_backbones/psmt_resnet50_identity.json')
    path=ROOT/info['path']
    if sha256(path)!=info['sha256']:raise RuntimeError('Author-posted ImageNet backbone changed')
    return {'path':info['path'],'sha256':info['sha256'],'author_link':info['author_link'],
        'role':'Author-posted ImageNet initialization, not EL-task-trained PV-S3 weights',
        'source_commit':'e062f81d602aea4cbae45c13ec85cfe79f95aa7e'}


def load_published_backbone(model,model_file,is_restore=False):
    if is_restore:raise ValueError('DDP prefix restoration is not part of this local port')
    identity=backbone_identity();path=Path(model_file)
    if path.resolve()!=(ROOT/identity['path']).resolve():raise PermissionError('Only the registered ImageNet backbone may initialize this port')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if 'model' in state:state=state['model']
    if 'state_dict' in state:state=state['state_dict']
    required=model.state_dict();missing=set(required)-set(state);extra=set(state)-set(required)
    if any(not key.endswith('num_batches_tracked') for key in missing):
        raise RuntimeError('Pretrained backbone has missing weights or normalization statistics: '+str(sorted(missing)))
    if not extra<= {'fc.weight','fc.bias'}:raise RuntimeError('Unexpected pretrained tensors: '+str(sorted(extra)))
    for key in set(required)&set(state):
        if state[key].shape!=required[key].shape or not torch.isfinite(state[key]).all():
            raise RuntimeError('Invalid pretrained tensor: '+key)
    loaded=model.load_state_dict({k:v for k,v in state.items() if k in required},strict=False)
    # PyTorch supplies legacy missing BN counters internally; learned tensors and
    # running mean/variance were already required above.
    if not set(loaded.missing_keys)<=missing or loaded.unexpected_keys:raise RuntimeError('Unexpected backbone load result')
    if any(int(model.state_dict()[key])!=0 for key in missing):raise RuntimeError('Legacy BN counters must initialize at zero')
    model.pretrained_load_audit={'loaded_tensors':len(set(required)&set(state)),
        'legacy_missing_batch_counters_initialized_at_zero':sorted(missing),'discarded_imagenet_classifier':sorted(extra),
        'weights_sha256':identity['sha256']}
    return model
