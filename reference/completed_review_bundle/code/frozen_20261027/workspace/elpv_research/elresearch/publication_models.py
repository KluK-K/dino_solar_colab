"""Official pretrained backbones with matched image-level classification heads."""
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F

from .adapt import inject_lora
from .common import sha256
from .encoders import FrozenEncoder


def build_backbone(name,size,device):
    if name=='dinov3_small':
        encoder=FrozenEncoder(name,size,str(device))
        identity={k:v for k,v in encoder.identity.items() if k!='device'}
        return encoder.model,encoder.model.config.hidden_size,identity
    from torchvision import models
    constructors={
        'resnet18':(models.resnet18,models.ResNet18_Weights.IMAGENET1K_V1,512),
        'resnet50':(models.resnet50,models.ResNet50_Weights.IMAGENET1K_V2,2048),
        'efficientnet_b0':(models.efficientnet_b0,models.EfficientNet_B0_Weights.IMAGENET1K_V1,1280)}
    constructor,weights,dimensions=constructors[name]
    model=constructor(weights=weights)
    if name.startswith('resnet'):model.fc=nn.Identity()
    else:model.classifier=nn.Identity()
    checkpoint=Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(weights.url).path).name
    identity={'name':name,'weights':str(weights),'weights_url':weights.url,
              'weights_sha256':sha256(checkpoint),'dimensions':dimensions,
              'input_size':size,'dtype':'float32',
              'preprocessing':'whole-image grayscale->RGB letterbox bicubic ImageNet normalization'}
    return model.eval().requires_grad_(False).to(device),dimensions,identity


def d4_view(images,code):
    if not 0<=int(code)<8:raise ValueError('D4 code must be 0..7')
    value=torch.rot90(images,int(code)%4,(-2,-1))
    return value.flip(-1).contiguous() if code>=4 else value.contiguous()


class PublicationClassifier(nn.Module):
    def __init__(self,encoder,dimensions,name,adaptation,seed=11):
        super().__init__()
        if adaptation not in {'frozen','lora','full'}:raise ValueError(adaptation)
        if adaptation=='lora' and name!='dinov3_small':raise ValueError('LoRA applies to the DINOv3 arm')
        self.encoder,self.encoder_name,self.adaptation=encoder,name,adaptation
        self.dimensions=dimensions
        self.encoder.requires_grad_(False)
        if adaptation=='lora':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+10000);inject_lora(self.encoder,rank=8,blocks=4)
        elif adaptation=='full':self.encoder.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.head=nn.Linear(dimensions,1)

    def train(self,mode=True):
        super().train(mode)
        if self.adaptation=='frozen' or self.encoder_name=='dinov3_small':self.encoder.eval()
        # Small annotation budgets: retain pretrained BN statistics, train affine weights.
        for module in self.encoder.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def features(self,images):
        if self.encoder_name=='dinov3_small':return self.encoder(pixel_values=images).last_hidden_state[:,0]
        return self.encoder(images)

    def classify(self,features):
        return self.head(F.layer_norm(features,(self.dimensions,))).squeeze(-1)

    def forward(self,images):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.adaptation!='frozen'):
            features=self.features(images)
        return self.classify(features)
