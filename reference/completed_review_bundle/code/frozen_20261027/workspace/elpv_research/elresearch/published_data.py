"""Geometry-aligned weak/strong views; unlabelled loading never reads annotation files."""
import math
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from .data import load_image
from .encoders import image_tensor
from .publication_models import d4_view


def prepare_unlabelled(rows,size):
    if any(r['partition']!='train' for r in rows):raise PermissionError('Unlabelled source pool must be source TRAINING only')
    items=[]
    for row in rows:
        image,g=image_tensor(load_image(row),size)
        valid=torch.zeros(size,size,dtype=torch.bool)
        valid[g['top']:g['top']+g['h'],g['left']:g['left']+g['w']]=True
        items.append({'id':row['id'],'image':image,'valid':valid,'geometry':g,'group':row['split_group']})
    return items


def weak_unlabelled_batch(items,codes,device):
    if any({'target','mask','native_mask','y'}&set(item) for item in items):raise PermissionError('Annotation entered unlabelled batch')
    return {'images':torch.stack([d4_view(x['image'],int(c)) for x,c in zip(items,codes)]).to(device),
        'valid':torch.stack([d4_view(x['valid'],int(c)) for x,c in zip(items,codes)]).to(device)}


def strong_photometric(images,rng):
    mean=torch.tensor([.485,.456,.406],device=images.device,dtype=images.dtype)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=images.device,dtype=images.dtype)[None,:,None,None]
    gray=(images*std+mean)[:,:1].clamp(0,1);outputs=[]
    for image in gray:
        value=image
        kernel=int(rng.random()*4.95);kernel=kernel+1 if kernel%2==0 else kernel
        if rng.random()<.8:
            brightness,contrast=rng.uniform(.5,1.5,2)
            # Grayscale EL has no hue/saturation; retain brightness/contrast perturbations.
            if rng.random()<.5:
                value=(value*brightness).clamp(0,1);value=((value-value.mean())*contrast+value.mean()).clamp(0,1)
            else:
                value=((value-value.mean())*contrast+value.mean()).clamp(0,1);value=(value*brightness).clamp(0,1)
        rng.random()  # Released RandomGrayscale(p.2) is a no-op on grayscale EL.
        if rng.random()<.5:
            value=gaussian_blur(value,[kernel,kernel],float(rng.uniform(.1,2.)))
        outputs.append(value.repeat(3,1,1))
    return (torch.stack(outputs)-mean)/std


def cutmix_unlabelled(images,pseudo_logits,valid,rng):
    if images.shape[0]!=pseudo_logits.shape[0] or valid.shape!=images.shape[:1]+images.shape[2:]:
        raise ValueError('CutMix geometry differs')
    mixed=images.clone();target=pseudo_logits.detach().clone();mask=valid.clone()
    batch,_,height,width=images.shape;donors=rng.permutation(batch)
    ratio=math.sqrt(1-float(rng.beta(4,4)));cut_h,cut_w=int(height*ratio),int(width*ratio)
    for i,j in enumerate(donors):
        cy=int(rng.integers(max(0,height//8),height));cx=int(rng.integers(max(0,width//8),width))
        y0,y1=np.clip([cy-cut_h//2,cy+cut_h//2],0,height);x0,x1=np.clip([cx-cut_w//2,cx+cut_w//2],0,width)
        mixed[i,:,y0:y1,x0:x1]=images[j,:,y0:y1,x0:x1]
        target[i,:,y0:y1,x0:x1]=pseudo_logits[j,:,y0:y1,x0:x1]
        mask[i,y0:y1,x0:x1]=valid[j,y0:y1,x0:x1]
    return mixed,target,mask
