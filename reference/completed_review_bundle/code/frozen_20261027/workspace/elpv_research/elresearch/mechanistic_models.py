"""Explicit patch/distance controls and correctly frozen post-adaptation encoders."""
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT,sha256,digest
from .encoders import FrozenEncoder,split_tokens
from .full_models import top_evidence
from .publication_dense_models import NormalizedEvidenceHead,make_dense_dino
from .publication_normal_bank import OrientationNormalBank
from .packed_normal_bank import build_packed_postadapt_bank


def patch_features(encoder,images,valid):
    hidden=encoder(pixel_values=images).last_hidden_state
    grid=images.shape[-1]//encoder.config.patch_size
    cls,patch=split_tokens(hidden,grid,encoder.config.num_register_tokens)
    keep=F.avg_pool2d(valid[:,None].float(),encoder.config.patch_size,encoder.config.patch_size)[:,0]>.999
    if not keep.flatten(1).any(1).all():raise ValueError('No completely image-covered patch')
    mean=(patch*keep[...,None]).sum((1,2))/keep.sum((1,2))[:,None]
    return cls,patch,mean


class LocalPatchDINO(nn.Module):
    """Frozen last-layer patch vectors -> 128-wide MLP; no decoder/normal references."""
    def __init__(self,encoder,seed=11):
        super().__init__();self.encoder=encoder.eval().requires_grad_(False)
        dim=encoder.config.hidden_size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.patch_head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU(),nn.Linear(128,3))
            self.image_head=NormalizedEvidenceHead(2*dim)

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,images,valid,geometries=None,query_groups=None):
        with torch.no_grad():cls,patch,mean=patch_features(self.encoder,images,valid)
        local=self.patch_head(patch).permute(0,3,1,2).contiguous()
        segmentation=F.interpolate(local,images.shape[-2:],mode='bilinear',align_corners=False)
        evidence=top_evidence(segmentation,valid,.05)
        image=self.image_head(torch.cat([cls,mean,evidence[:,None]],-1)).squeeze(-1)
        return {'segmentation':segmentation,'patch_logits':local,'image':image}


def global_normal_distance(query,items,query_group,device='cpu',per_frame=64,chunk=2048):
    """Exact nearest cosine distance within a deterministic capped normal memory."""
    if not isinstance(query_group,tuple) or len(query_group)!=2:raise ValueError('Expected acquisition family and D4 orientation')
    group,view=query_group
    if not 0<=int(view)<8 or per_frame<1 or chunk<1:raise ValueError('Invalid normal-memory configuration')
    candidates=[]
    for item in items:
        if item['group']==group:continue  # Never use own image or capture family.
        value=item['views'][view]['patch'].float()
        if not len(value):continue
        indices=torch.linspace(0,len(value)-1,min(per_frame,len(value))).round().long()
        candidates.append(value[indices])
    if not candidates:raise ValueError('No eligible normal acquisition family')
    bank=F.normalize(torch.cat(candidates),dim=-1)
    query=F.normalize(torch.as_tensor(query,dtype=torch.float32,device=device),dim=-1)
    shape=query.shape[:-1];query=query.reshape(-1,query.shape[-1])
    best=torch.full((len(query),),-1.,device=device)
    for start in range(0,len(bank),chunk):
        best=torch.maximum(best,(query@bank[start:start+chunk].to(device).T).max(1).values)
    return (1-best).clamp(0,2).reshape(shape)


class GlobalNormalDistance(nn.Module):
    """Mask-informed normal memory; cosine response is not a crack posterior."""
    def __init__(self,encoder,bank,cutoff=1.,scale=10.):
        super().__init__();self.encoder=encoder.eval().requires_grad_(False);self.bank=bank
        self.register_buffer('pixel_cutoff',torch.tensor(float(cutoff)))
        self.scale=float(scale)

    def train(self,mode=True):super().train(False);return self

    def load_calibration_state(self,state):
        if set(state)!={'pixel_cutoff'} or state['pixel_cutoff'].numel()!=1:
            raise ValueError('Unexpected distance calibration checkpoint')
        value=state['pixel_cutoff'].float()
        if not torch.isfinite(value).all() or not 0<=float(value)<=2:raise ValueError('Invalid cosine-distance cutoff')
        self.pixel_cutoff.copy_(value.to(self.pixel_cutoff.device))

    def forward(self,images,valid,geometries,query_groups):
        if len(query_groups)!=len(images):raise ValueError('Missing query acquisition identities')
        with torch.no_grad():
            _,patch,_=patch_features(self.encoder,images,valid)
            distance=torch.stack([global_normal_distance(p,self.bank.items,g,images.device)
                for p,g in zip(patch,query_groups)])
            native=F.interpolate(distance[:,None],images.shape[-2:],mode='bilinear',align_corners=False)[:,0]
            signed=(native-self.pixel_cutoff)*self.scale
            segmentation=torch.stack([-.5*signed,.5*signed,torch.full_like(signed,-30.)],1)
            top=torch.stack([d[v].topk(max(1,math.ceil(int(v.sum())*.05))).values.mean() for d,v in zip(native,valid)])
            image=(top-self.pixel_cutoff)*self.scale
        return {'segmentation':segmentation,'image':image,'raw_distance':native}


def build_mechanistic_system(spec,seed,size,device,training_rows=None,bank=None):
    architecture=spec['architecture']
    if architecture=='dinov3_postadapt':
        folder=(ROOT/spec['adapted_encoder_path']).resolve()
        if not folder.is_relative_to(ROOT):raise PermissionError('Adapted encoder must be local to this study')
        if sha256(folder/'model.safetensors')!=spec['adapted_weights_sha256']:
            raise RuntimeError('Merged source-adapted encoder changed')
        if sha256(folder/'config.json')!=spec['adapted_config_sha256']:
            raise RuntimeError('Merged source-adapted configuration changed')
        encoder=FrozenEncoder('dinov3_small',size,str(device),model_path=folder)
    elif architecture in {'dinov3_local_patch','dinov3_normal_distance'}:
        encoder=FrozenEncoder('dinov3_small',size,str(device))
    else:raise ValueError('Unregistered mechanism architecture')
    identity={k:v for k,v in encoder.identity.items() if k!='device'}
    if architecture=='dinov3_postadapt':
        # Weight identity must survive relocation of the reproducibility package.
        identity['model_id']=spec['adapted_encoder_path']
        identity['config']={**identity['config'],'_name_or_path':spec['adapted_encoder_path']}
    if spec.get('reference') and bank is None:
        if training_rows is None:raise ValueError('Normal memory requires selected source training rows')
        if architecture=='dinov3_postadapt':
            bank=build_packed_postadapt_bank(training_rows,encoder.model,size,identity,device,folder)
        else:bank=OrientationNormalBank.build(training_rows,encoder.model,size,identity,device)
    if bank is not None and digest(bank.encoder_identity)!=digest(identity):
        raise RuntimeError('A pre-adaptation or different-encoder normal bank cannot be reused')
    if architecture=='dinov3_local_patch':
        if bank is not None:raise ValueError('Patch baseline has no normal memory')
        model=LocalPatchDINO(encoder.model,seed)
    elif architecture=='dinov3_normal_distance':
        if bank is None:raise ValueError('Distance control requires a normal memory')
        model=GlobalNormalDistance(encoder.model,bank,spec.get('pixel_distance_cutoff',1.))
    else:
        model=make_dense_dino(encoder.model,'frozen',seed,bank,.1,.05)
    return model.to(device),identity,bank
