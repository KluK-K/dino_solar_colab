"""Complete dense networks: pretrained DeepLabv3 and multi-depth DINOv3 decoder."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adapt import inject_lora
from .encoders import split_tokens, patch_geometry
from .classification_heads import make_image_head

def top_evidence(logits, valid, fraction=.05):
    odds = logits[:,1] - torch.logsumexp(logits[:,[0,2]], dim=1)
    return torch.stack([x[v].topk(max(1,math.ceil(int(v.sum())*fraction))).values.mean()
                        for x,v in zip(odds,valid)])

class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming,outgoing,3,padding=1,bias=False), nn.GroupNorm(8,outgoing), nn.GELU(),
                         nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False), nn.GroupNorm(8,outgoing), nn.GELU())

class FullDINO(nn.Module):
    """Four transformer depths + matched residual slot + four detail-skip decoder stages."""
    def __init__(self, backbone, bank=None, adaptation='frozen', width=128, reference_neighbors=5,
                 geometry_weight=.1, top_fraction=.05, image_head='linear'):
        super().__init__()
        if adaptation not in ['frozen','lora','full']:
            raise ValueError(adaptation)
        if adaptation!='frozen' and bank is not None:
            raise ValueError('Adapt without references, then freeze/re-encode/rebuild the bank before reference training')
        if width != 128:
            raise ValueError('Current protocol fixes decoder width at 128')
        self.encoder = backbone
        self.encoder.requires_grad_(False)
        if adaptation=='lora':
            inject_lora(self.encoder,rank=8,blocks=4)
        elif adaptation=='full':
            self.encoder.requires_grad_(True)
        self.adaptation, self.bank = adaptation, bank
        self.reference_neighbors, self.geometry_weight = reference_neighbors, geometry_weight
        self.top_fraction = top_fraction
        dim = backbone.config.hidden_size
        self.dimensions, self.patch_size = dim, backbone.config.patch_size
        if self.patch_size != 16:
            raise ValueError('This four-stage decoder expects patch size 16')
        count=len(backbone.layer)
        self.depths=[round(count*x)-1 for x in [.25,.5,.75,1.]]
        self.projections=nn.ModuleList([nn.Conv2d(dim,width,1) for _ in self.depths])
        self.normal_projection=nn.Sequential(nn.Conv2d(2*dim+3,64,1),nn.GroupNorm(8,64),nn.GELU())
        self.fuse=ConvBlock(4*width+64,width)
        channels=[width,96,64,32,24]
        self.decoder=nn.ModuleList([ConvBlock(channels[i]+1,channels[i+1]) for i in range(4)])
        self.segmentation=nn.Conv2d(24,3,1)
        self.image_head=make_image_head(2*dim+1,image_head)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Deterministic pretrained layers; LoRA factors still receive gradients.
        return self

    def forward(self, images, valid, geometries, query_groups):
        captured={}; handles=[]
        for index in self.depths:
            handles.append(self.encoder.layer[index].register_forward_hook(
                lambda module, args, output, index=index: captured.__setitem__(index,output)))
        try:
            output=self.encoder(pixel_values=images)
        finally:
            for handle in handles:
                handle.remove()
        grid=images.shape[-1]//self.patch_size
        registers=self.encoder.config.num_register_tokens
        cls,patch=split_tokens(output.last_hidden_state,grid,registers)
        patch_valid=F.avg_pool2d(valid[:,None].float(),self.patch_size,self.patch_size)[:,0] > .999
        if not patch_valid.flatten(1).any(1).all():
            raise ValueError('No valid patch in an input')
        mean=(patch*patch_valid[...,None]).sum((1,2))/patch_valid.sum((1,2))[:,None]
        depths=[]
        for index,projection in zip(self.depths,self.projections):
            _,value=split_tokens(self.encoder.norm(captured[index]),grid,registers)
            depths.append(projection(value.permute(0,3,1,2).contiguous()))
        reference=[]
        for i,(g,group) in enumerate(zip(geometries,query_groups)):
            _,xy=patch_geometry(g,self.patch_size)
            coords=torch.from_numpy(xy).to(images.device)
            query=F.normalize(patch[i],dim=-1)
            if self.bank is None:
                values=torch.cat([query,torch.zeros_like(query),coords,torch.zeros(grid,grid,1,device=images.device)],-1)
            else:
                feature={'cls':cls[i].detach().cpu().numpy(),'patch':patch[i].detach().cpu().numpy(),'coords':xy}
                values=self.bank.residual(feature,query_group=group,topk=self.reference_neighbors,
                                          geometry_weight=self.geometry_weight,device=images.device).to(images.device)
            reference.append(values.permute(2,0,1))
        normal=self.normal_projection(torch.stack(reference))
        value=self.fuse(torch.cat(depths+[normal],dim=1))
        detail=images[:,:1]
        for block in self.decoder:
            value=F.interpolate(value,scale_factor=2,mode='bilinear',align_corners=False)
            skip=F.interpolate(detail,size=value.shape[-2:],mode='bilinear',align_corners=False)
            value=block(torch.cat([value,skip],dim=1))
        segmentation=self.segmentation(value)
        evidence=top_evidence(segmentation,valid,self.top_fraction)
        image=self.image_head(torch.cat([cls,mean,evidence[:,None]],dim=-1)).squeeze(-1)
        return {'segmentation':segmentation,'image':image}

class FullDeepLab(nn.Module):
    """Original torchvision DeepLabv3/ResNet-50 architecture, fine-tuned for three classes."""
    def __init__(self, pretrained=True, top_fraction=.05):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        self.network=deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                                       weights_backbone=None,aux_loss=True)
        self.network.classifier[-1]=nn.Conv2d(256,3,1)
        self.network.aux_classifier[-1]=nn.Conv2d(256,3,1)
        self.top_fraction=top_fraction
        self.image_head=nn.Linear(1,1)
        with torch.no_grad():
            self.image_head.weight.fill_(1);self.image_head.bias.zero_()

    def train(self,mode=True):
        super().train(mode)
        # Small local batches: freeze pretrained BN running statistics, keep affine parameters trainable.
        for module in self.network.modules():
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self,images,valid,geometries=None,query_groups=None):
        result=self.network(images)
        evidence=top_evidence(result['out'],valid,self.top_fraction)
        return {'segmentation':result['out'],'auxiliary':result.get('aux'),
                'image':self.image_head(evidence[:,None]).squeeze(-1)}

def dense_loss(output,target,image_y,class_weight):
    """Identical primary image + pixel CE + crack Dice losses for both dense architectures."""
    from .data import IGNORE
    valid=target!=IGNORE
    ce=F.cross_entropy(output['segmentation'],target,weight=class_weight,ignore_index=IGNORE)
    probability=output['segmentation'].softmax(1)[:,1]*valid
    truth=(target==1).float()
    intersection=(probability*truth).sum((1,2))
    dice=1-((2*intersection+1)/(probability.sum((1,2))+truth.sum((1,2))+1)).mean()
    loss=F.binary_cross_entropy_with_logits(output['image'],image_y)+ce+.5*dice
    if output.get('auxiliary') is not None:
        loss=loss+.4*F.cross_entropy(output['auxiliary'],target,weight=class_weight,ignore_index=IGNORE)
    return loss
