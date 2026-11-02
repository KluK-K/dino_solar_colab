"""Gradient, initialization, and checkpoint contracts; fixtures are not research scores."""
import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from elresearch.classification_heads import make_image_head
from elresearch.image_classification import EmbeddingClassifier, trainable_state, restore_trainable, assert_source_partitions
from elresearch.full_models import FullDINO, dense_loss
from elresearch.encoders import geometry


def backbone():
    return DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                                        intermediate_size=64, patch_size=16, num_register_tokens=4))


@pytest.mark.parametrize('adaptation', ['frozen', 'lora', 'full'])
def test_image_classifier_live_gradient_and_trainable_checkpoint(adaptation):
    model=EmbeddingClassifier(backbone(), 32, adaptation=adaptation, head='mlp128_32').train()
    images=torch.randn(2,3,32,32)
    logits=model(images)
    torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.])).backward()
    assert logits.shape==(2,)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.head.parameters())
    if adaptation=='frozen':
        assert all(p.grad is None for p in model.encoder.parameters())
    else:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    state=trainable_state(model)
    model.eval(); before=model(images).detach()
    with torch.no_grad():
        for p in model.head.parameters():p.add_(.1)
    restore_trainable(model,state)
    assert torch.allclose(model(images),before,atol=1e-6)


def test_paired_head_initialization_and_cached_live_equivalence():
    original=backbone().state_dict()
    models=[]
    for adaptation in ['frozen','lora','full']:
        encoder=backbone();encoder.load_state_dict(original)
        models.append(EmbeddingClassifier(encoder,32,adaptation=adaptation,head='mlp128_32',seed=11).eval())
    images=torch.randn(2,3,32,32)
    with torch.no_grad():
        results=[m(images) for m in models]
        cached=models[0].encoder(pixel_values=images).last_hidden_state[:,0]
        assert torch.allclose(models[0].classify(cached),results[0],atol=1e-6)
    assert all(torch.allclose(results[0],r,atol=1e-6) for r in results[1:])


def test_dense_mlp_head_keeps_pixel_supervision_and_checkpoint_layout():
    torch.manual_seed(11)
    linear=FullDINO(backbone())
    assert 'image_head.weight' in linear.state_dict()  # Legacy checkpoints retain their names.
    mlp=FullDINO(backbone(),image_head='mlp128_32').train()
    images=torch.randn(2,3,64,64);valid=torch.ones(2,64,64,dtype=torch.bool)
    output=mlp(images,valid,[geometry(64,64,64)]*2,['a','b'])
    target=torch.zeros(2,64,64,dtype=torch.long);target[:,20:22,20:30]=1
    dense_loss(output,target,torch.tensor([1.,1.]),torch.ones(3)).backward()
    assert output['segmentation'].shape==(2,3,64,64)
    assert all(p.grad is not None for p in mlp.image_head.parameters())
    assert mlp.segmentation.weight.grad is not None
    assert all(p.grad is None for p in mlp.encoder.parameters())


def test_source_guard_and_unknown_head():
    rows=[{'id':str(i),'split_group':str(i),'pixel_sha256':str(i),'image_sha256':str(i),
           'target':i%2,'partition':'train'} for i in range(2)]
    with pytest.raises(ValueError,match='Unexpected source partition'):
        assert_source_partitions(rows,rows)
    validation=[{**r,'partition':'validation'} for r in rows]
    with pytest.raises(ValueError,match='leakage'):
        assert_source_partitions(rows,validation)
    with pytest.raises(ValueError):make_image_head(32,'unknown')
