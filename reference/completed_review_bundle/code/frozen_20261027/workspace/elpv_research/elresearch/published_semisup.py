"""Device/class-generic full teacher/VAT/Conf-CE mechanisms for a documented EL port."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT,digest
from .published_weights import backbone_identity
from .published_psmt_architecture import EncoderNetwork,DecoderNetwork
from .full_models import top_evidence


class PublishedNetwork(nn.Module):
    def __init__(self,seed,size=336,pretrained=True):
        super().__init__();self.size=size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder=EncoderNetwork(3,nn.BatchNorm2d,back_bone=50,
                pretrained_model=str(ROOT/backbone_identity()['path']) if pretrained else None)
            self.decoder=DecoderNetwork(3,[size,size])

    def features(self,images):return self.encoder(images)
    def decode(self,features):return self.decoder(features,data_shape=[self.size,self.size])
    def forward(self,images):return self.decode(self.features(images))


def unit_direction(value):
    return value/(value.flatten(1).norm(dim=1)+1e-8).reshape(-1,*([1]*(value.ndim-1)))


def teacher_vat(features,teachers,xi=1e-6,epsilon=2.):
    """Teachers remain frozen; only the detached perturbation direction gets gradients."""
    if len(teachers)!=2 or xi<=0 or epsilon<0:raise ValueError('Invalid teacher-VAT configuration')
    decoders=[teacher.decoder for teacher in teachers];modes=[m.training for m in decoders]
    for decoder in decoders:decoder.eval()
    try:
        detached=features.detach()
        with torch.no_grad():probability=F.softmax(sum(t.decode(detached) for t in teachers)/2,dim=1)
        # CPU RNG matches the released helper, then transfer to the compute device.
        direction=unit_direction((torch.rand(detached.shape)-.5).to(detached.device)).requires_grad_()
        with torch.enable_grad():
            logits=sum(t.decode(detached+xi*direction) for t in teachers)/2
            divergence=F.kl_div(F.log_softmax(logits,dim=1),probability,reduction='batchmean')
            gradient=torch.autograd.grad(divergence,direction,only_inputs=True)[0]
        perturbation=unit_direction(gradient.detach())*epsilon
        if not torch.isfinite(perturbation).all():raise RuntimeError('Nonfinite teacher-VAT perturbation')
        diagnostics={'xi':xi,'epsilon':epsilon,'direction_gradient_norm':float(gradient.norm().detach().cpu()),
            'adversarial_norm_mean':float(perturbation.flatten(1).norm(dim=1).mean().detach().cpu()),
            'finite_difference_KL':float(divergence.detach().cpu())}
        return perturbation,diagnostics
    finally:
        for decoder,mode in zip(decoders,modes):decoder.train(mode)


def confidence_ce(student_logits,teacher_logits,valid,positive_threshold=.6,negative_threshold=0.):
    if student_logits.shape!=teacher_logits.shape or valid.shape!=student_logits.shape[:1]+student_logits.shape[2:]:
        raise ValueError('Pseudo-label/image geometry mismatch')
    if not 0<=positive_threshold<=1 or not 0<=negative_threshold<=1:raise ValueError('Invalid confidence threshold')
    target=teacher_logits.detach();probability=target.softmax(1);weight,label=probability.max(1)
    keep=(weight>=positive_threshold)&valid
    zero=student_logits.sum()*0
    positive=(F.cross_entropy(student_logits,label,reduction='none')*weight)[keep].mean() if keep.any() else zero
    negative=zero;regularizer=zero
    if negative_threshold>0:
        other=1-F.one_hot(label,num_classes=student_logits.shape[1]).permute(0,3,1,2).to(student_logits.dtype)
        selected=(probability<negative_threshold)&valid[:,None]
        matrix=-other*(1-student_logits.softmax(1)).clamp(1e-7,1).log()
        if selected.any():negative=matrix[selected].mean()
        if valid.any():regularizer=.5*student_logits.softmax(1).square().permute(0,2,3,1)[valid].mean()
    result=positive+negative+regularizer
    return result,{'confident_pixel_fraction':float(keep.sum().detach().cpu()/valid.sum().clamp_min(1).cpu()),
        'positive_loss':float(positive.detach().cpu()),'negative_loss':float(negative.detach().cpu()),
        'confidence_regularizer':float(regularizer.detach().cpu())}


def assisted_logits(first,second,active_teacher):
    if first.shape!=second.shape or active_teacher not in {1,2}:raise ValueError('Invalid teacher pair')
    core,assist=(first,second) if active_teacher==1 else (second,first)
    # Released five-class topk4 generalizes to all-but-one class, not all3 classes.
    keep=torch.zeros_like(assist,dtype=torch.bool).scatter_(1,assist.topk(assist.shape[1]-1,dim=1).indices,True)
    masked=torch.where(keep,assist,torch.zeros_like(assist))
    # Preserve the released zero-fill convention, including exact-zero selected logits.
    assisted=torch.where(masked==0,core,masked)
    return .5*(core+assisted)


@torch.no_grad()
def ema_update(teacher,student,alpha=.996):
    if not 0<=alpha<=1:raise ValueError('Invalid EMA coefficient')
    source=student.state_dict();target=teacher.state_dict()
    if set(source)!=set(target):raise RuntimeError('EMA architecture differs')
    for key,value in target.items():
        if value.is_floating_point():value.mul_(alpha).add_(source[key],alpha=1-alpha)
        else:value.copy_(source[key])  # Batch counters are integers, never fractional EMA.


class PublishedTrainSystem(nn.Module):
    def __init__(self,seed,size,semi):
        super().__init__();self.seed,self.size,self.semi=seed,size,semi
        self.student=PublishedNetwork(seed+3000,size,True)
        if semi:
            # Match the released teacher-random/student-ImageNet role policy.
            self.first=PublishedNetwork(seed+1000,size,False).requires_grad_(False)
            self.second=PublishedNetwork(seed+2000,size,False).requires_grad_(False)

    def supervised(self,images,use_vat=True,xi=1e-6):
        features=self.student.features(images);diagnostics=None
        if self.semi and use_vat:
            noise,diagnostics=teacher_vat(features,[self.first,self.second],xi=xi)
            features=features+noise  # No in-place modification of cached encoder features.
        return self.student.decode(features),diagnostics

    @torch.no_grad()
    def pseudo(self,images,active_teacher):
        return assisted_logits(self.first(images),self.second(images),active_teacher)


class PublishedInference(nn.Module):
    """Storage basis is the reproducible student initialization, not teacher training init."""
    def __init__(self,seed,size,semi,view='primary'):
        super().__init__();self.semi,self.view=semi,view
        self.first=PublishedNetwork(seed+3000,size,True)
        if semi:self.second=PublishedNetwork(seed+3000,size,True)
        self.eval()

    def forward(self,images,valid,geometries=None,query_groups=None):
        logits=self.first(images)
        if self.semi and self.view=='ensemble':logits=.5*(logits+self.second(images))
        elif self.view not in {'primary','teacher1'}:raise ValueError('Unregistered inference view')
        return {'segmentation':logits,'image':top_evidence(logits,valid,.05)}


def published_identity(size,semi):
    return {'name':'PSMT/PV-S3 deep-stem ResNet50 DeepLabv3+ canonical3-class port',
        'architecture_commit':'058cdc3bd8fe5faa83dbc798cfb4c136b74207fd','backbone':backbone_identity(),
        'input_size':size,'classes':3,'semi':semi,'dtype':'float32',
        'primary_inference':'teacher1 as released PV-S3 inference.py' if semi else 'supervised student',
        'extra_view':'teacher-logit ensemble at the same source-selected checkpoint' if semi else None,
        'limits':'Architecture/mechanisms preserved; canonical labels, preprocessing, device, batch/updates and validation differ from the published experiment. Not a reproduction of its reported score.'}


def inference_state(system):
    if system.semi:
        return {'first.'+k:v.detach().cpu() for k,v in system.first.state_dict().items()} | {
            'second.'+k:v.detach().cpu() for k,v in system.second.state_dict().items()}
    return {'first.'+k:v.detach().cpu() for k,v in system.student.state_dict().items()}
