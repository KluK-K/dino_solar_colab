"""Exact portable inference archives, including teacher BN state and tensor-basis proof."""
from pathlib import Path
import torch

from .common import ROOT,read_json,save_json,sha256,now,digest
from .image_classification import atomic_torch_save
from .published_semisup import PublishedNetwork,PublishedInference,published_identity
from .published_weights import backbone_identity
from .lossless_state import state_digest,save_lossless_state,load_lossless_state


def ensure_head_basis(seed,size=336):
    path=ROOT/'models/published_backbones'/f'psmt_head_basis_seed{seed}_classes3.pt';metadata=path.with_suffix('.json')
    if metadata.exists():
        binding=read_json(metadata)
        if sha256(path)!=binding['sha256'] or binding['backbone_sha256']!=backbone_identity()['sha256']:
            raise RuntimeError('Exact archive head basis changed')
        return binding
    model=PublishedNetwork(seed+3000,size,True)
    heads={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('encoder.backbone.')}
    atomic_torch_save(heads,path)
    binding={'at':now(),'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'seed':seed,
        'backbone_sha256':backbone_identity()['sha256'],'head_tensor_sha256':state_digest(heads),
        'architecture_commit':'058cdc3bd8fe5faa83dbc798cfb4c136b74207fd',
        'purpose':'Exact storage basis, preserving inference across hosts without relying on re-generated random heads. Not an additional trained model.'}
    save_json(metadata,binding);return binding


def install_basis_heads(network,binding):
    path=ROOT/binding['path']
    if sha256(path)!=binding['sha256']:raise RuntimeError('Archive head basis changed')
    heads=torch.load(path,map_location='cpu',weights_only=True)
    expected={k for k in network.state_dict() if not k.startswith('encoder.backbone.')}
    if set(heads)!=expected or state_digest(heads)!=binding['head_tensor_sha256']:
        raise RuntimeError('Archive head basis schema changed')
    result=network.load_state_dict(heads,strict=False)
    if result.unexpected_keys or any(not k.startswith('encoder.backbone.') for k in result.missing_keys):
        raise RuntimeError('Unexpected archive basis load')


def inference_basis(seed,size,semi,binding=None):
    binding=binding or ensure_head_basis(seed,size)
    model=PublishedInference(seed,size,semi)
    install_basis_heads(model.first,binding)
    if semi:install_basis_heads(model.second,binding)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return model,state,binding


def load_published_inference(run,checkpoint,device='cpu',view='primary'):
    spec=run['model'];binding=spec['archive_basis'];semi=spec['semi']
    if spec['backbone_sha256']!=backbone_identity()['sha256']:raise RuntimeError('Published-method initialization changed')
    model,basis,actual=inference_basis(run['seed'],run['input_size'],semi,binding)
    state,record=load_lossless_state(checkpoint,basis,binding)
    model.load_state_dict(state,strict=True);model.view=view
    if digest(published_identity(run['input_size'],semi))!=digest(run['encoder']):raise RuntimeError('Published encoder identity changed')
    return model.eval().requires_grad_(False).to(device)
