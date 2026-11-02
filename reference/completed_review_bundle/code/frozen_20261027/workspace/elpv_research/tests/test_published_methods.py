import ast
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from elresearch.common import ROOT,sha256
from elresearch.lossless_state import save_lossless_state,load_lossless_state,state_digest
from elresearch.published_semisup import teacher_vat,confidence_ce,ema_update,assisted_logits
from elresearch.published_data import prepare_unlabelled,weak_unlabelled_batch,cutmix_unlabelled


def test_lossless_storage_restores_all_float_bits_and_bn_buffers(tmp_path):
    basis={'weight':torch.randn(3,4),'running_mean':torch.zeros(3),'running_var':torch.ones(3),
        'num_batches_tracked':torch.tensor(0,dtype=torch.long),'signed_zero':torch.tensor([0.])}
    state={k:v.clone() for k,v in basis.items()};state['weight']+=.001;state['running_mean']+=.3
    state['running_var']*=.7;state['num_batches_tracked'].fill_(17);state['signed_zero'].fill_(-0.)
    binding={'path':'fixture','sha256':'fixed'};path=tmp_path/'best.pt'
    info=save_lossless_state(path,state,basis,binding);restored,record=load_lossless_state(path,basis,binding)
    assert state_digest(restored)==state_digest(state)==info['state_tensor_sha256']
    assert int(restored['num_batches_tracked'])==17 and np.signbit(restored['signed_zero'].numpy())[0]
    wrong={k:v.clone() for k,v in basis.items()};wrong['weight'][0,0]+=1
    with pytest.raises(RuntimeError,match='basis tensors'):load_lossless_state(path,wrong,binding)
    with pytest.raises(RuntimeError,match='binding'):load_lossless_state(path,basis,{'path':'changed'})


def test_confidence_loss_matches_released_five_class_math():
    # Execute only the reviewed loss function as a numerical oracle. Its CUDA-padding
    # path is not entered because the five-class fixture contains class4.
    path=ROOT/'references/publication_20261017/PV-S3_upstream_code/main/Utils/losses.py'
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='semi_ce_loss')
    namespace={'torch':torch,'F':F};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    teacher=torch.full((1,5,2,2),-3.);teacher[:,4]=4.
    student=torch.randn_like(teacher,requires_grad=True);valid=torch.ones(1,2,2,dtype=torch.bool)
    for negative in [0.,.8]:
        positive,_,neg=namespace['semi_ce_loss'](student,teacher,threshold=.6,threshold_neg=negative)
        reference=positive+(neg+.5*student.softmax(1).square().mean() if negative else 0)
        ours,_=confidence_ce(student,teacher,valid,.6,negative)
        torch.testing.assert_close(ours,reference)


def test_three_class_empty_confidence_and_geometry_masks_have_safe_gradients():
    student=torch.randn(2,3,4,4,requires_grad=True);teacher=torch.zeros_like(student)
    valid=torch.zeros(2,4,4,dtype=torch.bool);valid[:,:2]=True
    loss,_=confidence_ce(student,teacher,valid,.6,0.);loss.backward()
    assert loss.item()==0 and torch.equal(student.grad,torch.zeros_like(student))
    teacher=torch.randn_like(student)*4
    first,_=confidence_ce(student,teacher,valid,.6,.8)
    altered=student.detach().clone();altered[:,:,2:]=1000*torch.randn_like(altered[:,:,2:])
    second,_=confidence_ce(altered,teacher,valid,.6,.8)
    torch.testing.assert_close(first,second)


def test_teacher_vat_does_not_backpropagate_to_teacher_or_encoder_and_restores_mode():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__();self.decoder=nn.Sequential(nn.Conv2d(4,3,1),nn.BatchNorm2d(3))
        def decode(self,x):return self.decoder(x)
    teachers=[Teacher().requires_grad_(False).train(),Teacher().requires_grad_(False).train()]
    features=torch.randn(2,4,5,5,requires_grad=True)
    perturbation,info=teacher_vat(features,teachers,xi=.1)
    assert features.grad is None and not perturbation.requires_grad
    assert all(t.decoder.training and all(p.grad is None for p in t.parameters()) for t in teachers)
    assert 1.<info['adversarial_norm_mean']<=2.00001 and torch.isfinite(perturbation).all()


def test_ema_copies_integer_counters_and_averages_all_float_buffers():
    teacher=nn.BatchNorm2d(2);student=nn.BatchNorm2d(2)
    with torch.no_grad():
        teacher.weight.fill_(0.);student.weight.fill_(2.);student.running_mean.fill_(4.);student.num_batches_tracked.fill_(9)
    ema_update(teacher,student,.75)
    torch.testing.assert_close(teacher.weight,torch.full((2,),.5))
    torch.testing.assert_close(teacher.running_mean,torch.ones(2));assert int(teacher.num_batches_tracked)==9


def test_unlabelled_source_loader_never_reads_hidden_masks_or_targets(tmp_path):
    image=tmp_path/'source.png';Image.fromarray(np.arange(32*48,dtype=np.uint8).reshape(32,48)).save(image)
    row={'id':'source','partition':'train','image_path':str(image),'image_sha256':sha256(image),
        'split_group':'source_group','target':0,'mask_path':'nonexistent_mask'}
    first=prepare_unlabelled([row],64);second=prepare_unlabelled([{**row,'target':1,'mask_path':'another nonexistent file'}],64)
    torch.testing.assert_close(first[0]['image'],second[0]['image'])
    assert not {'target','mask','native_mask','y'}&set(first[0])
    with pytest.raises(PermissionError):prepare_unlabelled([{**row,'partition':'test'}],64)
    with pytest.raises(PermissionError):weak_unlabelled_batch([{**first[0],'target':0}],[0],'cpu')


def test_cutmix_pseudo_labels_and_validity_follow_the_same_donor_pixels():
    images=torch.stack([torch.zeros(3,16,16),torch.ones(3,16,16)])
    logits=torch.stack([torch.zeros(3,16,16),torch.ones(3,16,16)*7])
    valid=torch.stack([torch.zeros(16,16,dtype=torch.bool),torch.ones(16,16,dtype=torch.bool)])
    output,pseudo,mask=cutmix_unlabelled(images,logits,valid,np.random.default_rng(7))
    torch.testing.assert_close(pseudo,output*7)
    assert torch.equal(mask,output[:,0].bool())


def test_assistance_keeps_one_class_from_core_for_three_class_port():
    first=torch.tensor([[[[10.]],[[20.]],[[30.]]]])
    second=torch.tensor([[[[3.]],[[2.]],[[1.]]]])
    value=assisted_logits(first,second,1)
    torch.testing.assert_close(value,torch.tensor([[[[6.5]],[[11.]],[[30.]]]]))


@pytest.mark.parametrize('source,local',[
    ('VocCode/Model/Deeplabv3_plus/encoder_decoder.py','published_psmt_architecture.py'),
    ('VocCode/Model/Deeplabv3_plus/Backbones/resnet.py','published_psmt_resnet.py')])
def test_full_network_bodies_match_mit_upstream_including_unused_source_layers(source,local):
    class Compatibility(ast.NodeTransformer):
        def visit_Compare(self,node):
            # The sole numerical-comparison compatibility correction is explicit.
            if (isinstance(node.left,ast.Name) and node.left.id=='back_bone'
                and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant)
                and node.comparators[0].value==101):node.ops=[ast.Eq()]
            return self.generic_visit(node)
    def canonical(path):
        tree=Compatibility().visit(ast.parse(path.read_text()))
        tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))
                   and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        return ast.dump(tree,include_attributes=False)
    assert canonical(ROOT/'references/publication_20261017/PS-MT_upstream_code'/source)==canonical(ROOT/'elresearch'/local)
