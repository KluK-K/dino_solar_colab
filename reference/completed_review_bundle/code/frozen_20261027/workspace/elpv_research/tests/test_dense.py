"""Complete network shape, supervision alignment and parameter-gradient contracts."""
import numpy as np
import torch
from transformers import DINOv3ViTModel,DINOv3ViTConfig
from elresearch.full_models import FullDINO,top_evidence,dense_loss
from elresearch.dense_train import dense_target
from elresearch.encoders import geometry

def test_full_decoder_output_and_gradient_contract():
    torch.manual_seed(11)
    backbone=DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32,num_hidden_layers=4,num_attention_heads=4,
                                          intermediate_size=64,patch_size=16,num_register_tokens=4))
    model=FullDINO(backbone,adaptation='frozen').train()
    image=torch.randn(2,3,64,64);valid=torch.ones(2,64,64,dtype=torch.bool)
    output=model(image,valid,[geometry(64,64,64)]*2,['A','B'])
    assert output['segmentation'].shape==(2,3,64,64) and output['image'].shape==(2,)
    target=torch.zeros(2,64,64,dtype=torch.long);target[:,20:24,30:34]=1
    dense_loss(output,target,torch.ones(2),torch.ones(3)).backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert model.segmentation.weight.grad is not None
    assert all(layer.weight.grad is not None for layer in model.projections)

def test_dense_alignment_and_thin_annotation():
    mask=np.zeros((600,300),np.uint8);mask[303,153]=1
    g=geometry(300,600,224);target=dense_target(mask,g)
    assert target.shape==(224,224) and (target==1).any()
    assert (target[:,:g['left']]==255).all()

def test_top_evidence_ignores_padding():
    values=torch.zeros(1,3,4,4);valid=torch.zeros(1,4,4,dtype=torch.bool);valid[:,:2,:2]=True
    before=top_evidence(values,valid)
    values[:,1,2:,:]=10000
    assert torch.allclose(before,top_evidence(values,valid))
