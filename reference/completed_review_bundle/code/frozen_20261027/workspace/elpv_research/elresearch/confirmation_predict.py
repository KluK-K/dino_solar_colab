"""Frozen neural/classical scorers with no fitting and no annotation input."""
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from PIL import Image

from .common import ROOT, read_json, sha256, digest, choose_device
from .classical_features import hog_statistics
from .confirmation_registry import check_file,verify_source_implementation
from .image_classification import restore_trainable
from .publication_models import build_backbone, PublicationClassifier
from .publication_dense_train import build_system
from .publication_normal_bank import OrientationNormalBank
from .representation_features import FrozenRepresentationEncoder,normalized_representation
from .mechanistic_models import build_mechanistic_system
from .mechanistic_evaluate import restore_mechanistic
from .packed_normal_bank import load_reference_bank
from .published_archive import load_published_inference


def _check_source_code(protocol,run):
    return verify_source_implementation(protocol,run)


class FrozenScorer:
    def __init__(self,binding,device='auto'):
        self.binding=binding;self.device=choose_device(device)
        self.run=read_json(check_file(binding['run']))
        protocol=read_json(check_file(binding['source_protocol']))
        _check_source_code(protocol,self.run)
        self.size=protocol.get('input_size',336)
        self.batch_size=protocol.get('batch_size',8 if binding['kind']!='dense' else 2)
        self.encoder=None;self.model=None;self.dimensions=None;self.representation_pooling=None
        checkpoint=check_file(binding['checkpoint'])
        if binding['kind']=='image':
            spec=self.run['model']
            encoder,dimensions,identity=build_backbone(spec['encoder'],self.size,self.device)
            if digest(identity)!=digest(self.run['encoder']):raise RuntimeError('Selected pretrained encoder changed')
            self.model=PublicationClassifier(encoder,dimensions,spec['encoder'],spec['adaptation'],self.run['seed']).to(self.device)
            restore_trainable(self.model,torch.load(checkpoint,map_location='cpu',weights_only=True))
            self.cached=self.run['cached_augmented_features']
            if self.cached:self.model.head.to('cpu')
            self.model.eval()
        elif binding['kind']=='dense':
            architecture=self.run['model']['architecture']
            if architecture=='published_psmt_deeplab':
                view=binding.get('source_view','primary')
                # Validate/decode the complete source archive on CPU. Primary
                # teacher1 inference needs no resident second teacher on MPS.
                self.model=load_published_inference(self.run,checkpoint,'cpu',view)
                if self.run['model'].get('semi') and view=='primary':del self.model.second
                self.model.to(self.device)
            else:
                recipe=binding.get('normal_bank_recipe')
                bank=load_reference_bank(check_file(recipe),self.run['training_ids']) if recipe else None
                mechanism=architecture in {'dinov3_local_patch','dinov3_normal_distance','dinov3_postadapt'}
                builder=build_mechanistic_system if mechanism else build_system
                self.model,identity,_=builder(self.run['model'],self.run['seed'],self.size,self.device,bank=bank)
                if digest(identity)!=digest(self.run['encoder']):raise RuntimeError('Selected dense encoder changed')
                if mechanism:restore_mechanistic(self.model,self.run['model'],checkpoint)
                else:restore_trainable(self.model,torch.load(checkpoint,map_location='cpu',weights_only=True))
            self.model.eval()
        elif binding['kind']=='classical':
            # Only this locally trained, checksum-bound estimator is loaded.
            self.model=joblib.load(checkpoint)
            identity=self.run['feature_identity'];self.dimensions=self.run['dimensions']
            if identity['kind']=='dinov3_cls':
                self.encoder,dimensions,actual=build_backbone('dinov3_small',self.size,self.device)
                if dimensions!=self.dimensions or digest(actual)!=digest(identity['encoder']):
                    raise RuntimeError('Frozen classical-probe encoder changed')
                self.encoder.eval().requires_grad_(False)
            elif identity['kind']=='frozen_representation_v1':
                self.size=identity['input_size'];self.batch_size=protocol.get('feature_batch_size',4)
                self.encoder=FrozenRepresentationEncoder(identity['encoder'],self.size,self.device)
                self.representation_pooling=identity['pooling']
            elif identity['kind']!='hog_statistics':raise ValueError('Unimplemented frozen feature identity')
        else:raise ValueError(binding['kind'])
        if isinstance(self.model,torch.nn.Module) and self.model.training:raise RuntimeError('Scorer must remain in evaluation mode')

    def __call__(self,batch):
        # The streaming engine supplies images/geometry only, never class labels/masks.
        if {'target','targets','mask','masks','y'} & set(batch):raise PermissionError('Annotations must not enter the model scorer')
        with torch.inference_mode():
            if self.binding['kind']=='image':
                images=batch['images'].to(self.device)
                logits=(self.model.classify(self.model.features(images).cpu()) if self.cached else self.model(images))
                return {'image_logits':logits.cpu()}
            if self.binding['kind']=='dense':
                out=self.model(batch['images'].to(self.device),batch['valid'].to(self.device),
                               batch['geometries'],[(g,0) for g in batch['groups']])
                return {'image_logits':out['image'].cpu(),'crack_probability':out['segmentation'].softmax(1)[:,1].cpu()}
            if self.representation_pooling is not None:
                raw_cls,raw_mean=self.encoder.encode([Image.fromarray(np.asarray(g,dtype=np.uint8)) for g in batch['gray_images']],views=[0])
                features=normalized_representation(raw_cls,raw_mean,self.representation_pooling)[:,0].astype(np.float64)
                if features.shape[-1]!=self.dimensions:raise RuntimeError('Frozen probe feature dimensions changed')
            elif self.encoder is None:
                features=np.stack([hog_statistics(g) for g in batch['gray_images']]).astype(np.float64)
            else:
                raw=self.encoder(pixel_values=batch['images'].to(self.device)).last_hidden_state[:,0].cpu()
                features=F.layer_norm(raw,(self.dimensions,)).numpy().astype(np.float64)
            with threadpool_limits(limits=2):logits=self.model.decision_function(features)
            return {'image_logits':np.asarray(logits,dtype=np.float64)}
