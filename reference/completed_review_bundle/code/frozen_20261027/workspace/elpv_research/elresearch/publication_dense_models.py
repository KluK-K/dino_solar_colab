"""Matched dense-supervision systems; separate from the immutable UCF pilot."""
import torch
from torch import nn
from torch.nn import functional as F

from .adapt import inject_lora
from .full_models import FullDINO, ConvBlock, top_evidence


class NormalizedEvidenceHead(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.linear = nn.Linear(dimensions + 1, 1)

    def forward(self, value):
        features = F.layer_norm(value[..., :self.dimensions], (self.dimensions,))
        return self.linear(torch.cat([features, value[..., -1:]], -1))


def make_dense_dino(backbone, adaptation, seed, bank=None, geometry_weight=.1,
                    top_fraction=.05):
    if adaptation not in {'frozen', 'lora', 'full'}:
        raise ValueError(adaptation)
    if bank is not None and adaptation != 'frozen':
        raise ValueError('A moving encoder cannot use a stale normal-reference bank')
    # Initialize all decoder/image-head parameters before independently injecting LoRA.
    # Frozen, reference, LoRA and full arms therefore start from identical new heads.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FullDINO(backbone, bank, 'frozen', geometry_weight=geometry_weight,
                         top_fraction=top_fraction)
        model.image_head = NormalizedEvidenceHead(2 * backbone.config.hidden_size)
    if adaptation == 'lora':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 10000)
            inject_lora(model.encoder, rank=8, blocks=4)
    elif adaptation == 'full':
        model.encoder.requires_grad_(True)
    model.adaptation = adaptation
    return model


def group_norm_heads(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(8, child.num_features)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            group_norm_heads(child)
    return module


class PublicationDeepLabPlus(nn.Module):
    """ResNet50 OS16 + ASPP + low-level skip; adapted baseline, not PV-S3 replication.

    ImageNet-V2 backbone; new heads use GN; pretrained backbone BN statistics stay
    frozen for batch 2. No teacher, pseudo-labels or extra unlabelled images.
    """
    def __init__(self, backbone, seed=11, top_fraction=.05):
        super().__init__()
        from torchvision.models.segmentation.deeplabv3 import ASPP
        self.encoder = backbone
        self.encoder.requires_grad_(True)
        self.encoder.fc = nn.Identity()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.aspp = group_norm_heads(ASPP(2048, [6, 12, 18], 256))
            self.low = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                     nn.GroupNorm(8, 48), nn.GELU())
            self.decode = ConvBlock(304, 128)
            self.segment = nn.Conv2d(128, 3, 1)
            self.image_head = NormalizedEvidenceHead(2048)
        self.top_fraction = top_fraction

    def train(self, mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images, valid, geometries=None, query_groups=None):
        e = self.encoder
        value = e.maxpool(e.relu(e.bn1(e.conv1(images))))
        low = e.layer1(value)
        high = e.layer4(e.layer3(e.layer2(low)))
        value = F.interpolate(self.aspp(high), low.shape[-2:], mode='bilinear', align_corners=False)
        value = self.segment(self.decode(torch.cat([value, self.low(low)], 1)))
        segmentation = F.interpolate(value, images.shape[-2:], mode='bilinear', align_corners=False)
        evidence = top_evidence(segmentation, valid, self.top_fraction)
        # Mask padded image area in the pooled global branch, independently of labels.
        keep = F.adaptive_avg_pool2d(valid[:, None].float(), high.shape[-2:])
        pooled = (high * keep).sum((2, 3)) / keep.sum((2, 3)).clamp_min(1)
        image = self.image_head(torch.cat([pooled, evidence[:, None]], -1)).squeeze(-1)
        return {'image': image, 'segmentation': segmentation}
