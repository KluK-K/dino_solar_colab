"""Guard scientific contracts of the next experiment; fixtures are not research results."""
import copy

import numpy as np
import pytest
import torch
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from elresearch.image_classification import restore_trainable, trainable_state
from elresearch.publication_models import PublicationClassifier, d4_view
from elresearch.publication_train import budget_rows, calibration, probability_metrics, round_plan


def tiny_dino():
    return DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=64, patch_size=16, num_register_tokens=4))


def test_d4_preserves_full_image_and_has_eight_distinct_views():
    image=torch.arange(3*4*4).reshape(3,4,4)
    views=[d4_view(image,i) for i in range(8)]
    assert len({v.numpy().tobytes() for v in views})==8
    for view in views:
        assert torch.equal(view.flatten().sort().values,image.flatten().sort().values)
        assert view.is_contiguous()
    with pytest.raises(ValueError):d4_view(image,8)


def test_annotation_budgets_are_nested_and_batch_plan_is_reproducible():
    rows=[{'id':str(i),'target':i%2} for i in range(60)]
    previous=set()
    for fraction in [.05,.1,.2,1.]:
        selected=budget_rows(rows,fraction,11)
        assert previous<={r['id'] for r in selected}
        assert {r['target'] for r in selected}=={0,1}
        previous={r['id'] for r in selected}
    first=round_plan(13,11,2,50,8)
    repeated=round_plan(13,11,2,50,8)
    assert len(first[0])==400 and set(first[0])==set(range(13))
    assert all(np.array_equal(a,b) for a,b in zip(first,repeated))
    assert not np.array_equal(first[0],round_plan(13,11,3,50,8)[0])


@pytest.mark.parametrize('adaptation',['frozen','lora','full'])
def test_live_and_cached_d4_gradients_and_checkpoint(adaptation):
    model=PublicationClassifier(tiny_dino(),32,'dinov3_small',adaptation).eval()
    image=torch.randn(2,3,32,32)
    for code in range(8):
        view=d4_view(image,code)
        with torch.no_grad():
            assert torch.allclose(model(view),model.classify(model.features(view)),atol=1e-6)
    model.train()
    torch.nn.functional.binary_cross_entropy_with_logits(model(image),torch.tensor([0.,1.])).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.head.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters()) == (adaptation!='frozen')
    state=trainable_state(model);model.eval();before=model(image).detach()
    with torch.no_grad():model.head.bias.add_(1.)
    restore_trainable(model,state)
    assert torch.allclose(before,model(image),atol=1e-6)


def test_full_cnn_trains_features_but_preserves_pretrained_bn_buffers():
    backbone=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.BatchNorm2d(8),nn.ReLU(),
                           nn.AdaptiveAvgPool2d(1),nn.Flatten())
    initial=copy.deepcopy(backbone.state_dict())
    model=PublicationClassifier(backbone,8,'resnet18','full').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    loss=model(torch.randn(2,3,16,16)).square().sum()
    loss.backward();optimizer.step()
    assert not torch.equal(backbone[0].weight,initial['0.weight'])
    assert not torch.equal(backbone[1].weight,initial['1.weight'])
    for name in ['running_mean','running_var','num_batches_tracked']:
        assert torch.equal(getattr(backbone[1],name),initial['1.'+name])
    assert set(trainable_state(model))=={n for n,p in model.named_parameters() if p.requires_grad}


def test_temperature_calibration_keeps_ranking_and_reduces_source_nll():
    y=np.array([0,0,0,1,0,1,1,1]);logits=np.array([-12.,-8.,-5.,-1.,1.,5.,8.,12.])
    temperature=calibration(y,logits)
    raw=probability_metrics(y,logits)
    adjusted=probability_metrics(y,logits,temperature)
    assert .05<=temperature<=20
    assert adjusted['negative_log_likelihood']<=raw['negative_log_likelihood']
    np.testing.assert_array_equal(np.argsort(logits),np.argsort(logits/temperature))
    assert np.isfinite(list(adjusted.values())).all()


@pytest.mark.parametrize('adaptation',['frozen','lora','full'])
def test_training_checkpoint_reproduces_saved_source_logits(tmp_path,monkeypatch,adaptation):
    from elresearch import publication_train as trainer
    torch.manual_seed(7);initial=copy.deepcopy(tiny_dino().state_dict())
    def build(*args):
        model=tiny_dino();model.load_state_dict(initial)
        return model,32,{'fixture':'tiny; not a research encoder'}
    def images(rows,size):
        return torch.stack([torch.full((3,size,size),float(r['target']))+
                            torch.arange(size*size).reshape(1,size,size)/500 for r in rows])
    hashes=trainer.scientific_hashes()
    monkeypatch.setattr(trainer,'ROOT',tmp_path)
    monkeypatch.setattr(trainer,'scientific_hashes',lambda:hashes)
    monkeypatch.setattr(trainer,'build_backbone',build)
    monkeypatch.setattr(trainer,'prepare_images',images)
    rows=[{'id':str(i),'split_group':str(i),'pixel_sha256':str(i),'image_sha256':str(i),
           'partition':'train' if i<4 else 'validation','target':i%2} for i in range(8)]
    settings={'study_id':'fixture','scientific_hashes':hashes,'device':'cpu','input_size':32,
        'batch_size':2,'updates_per_round':1,'maximum_rounds':2,'patience':2,'weight_decay':1e-4,
        'target_source_recall':.95,'scope':'unit-test fixture only'}
    spec={'name':adaptation,'encoder':'dinov3_small','adaptation':adaptation,'head_lr':1e-3,'encoder_lr':1e-4}
    folder=tmp_path/'runs'/adaptation
    result=trainer.train(settings,spec,rows[:4],rows[4:],11,1.,folder)
    model=PublicationClassifier(build()[0],32,'dinov3_small',adaptation,11).eval()
    restore_trainable(model,torch.load(folder/'best.pt',weights_only=True))
    with torch.no_grad():reloaded=model(images(rows[4:],32)).numpy()
    with np.load(folder/'validation_predictions.npz') as saved:
        np.testing.assert_allclose(reloaded,saved['logits'],atol=1e-6)
    assert result['training_images']==4 and result['total_labeled_images_including_validation']==8
    assert trainer.train(settings,spec,rows[:4],rows[4:],11,1.,folder)==result


def test_real_development_folds_never_include_sealed_test_ids():
    from elresearch.common import ROOT,read_json
    from elresearch.publication_splits import partition_manifest,fold_records
    manifest=partition_manifest()
    sealed={r['id'] for r in read_json(ROOT/'data/scdd_publication_v1/manifest.json')['records']
            if r['partition']=='test'}
    assert len(sealed)==117
    for fold in manifest['folds']:
        train,validation,target=fold_records(manifest,fold['name'])
        assert not sealed&{r['id'] for r in train+validation+target}
        if target:
            assert {r['acquisition_source'] for r in target}=={fold['held_out_acquisition_source']}
            assert {r['acquisition_source'] for r in train+validation}.isdisjoint({fold['held_out_acquisition_source']})


def test_development_inference_rejects_test_and_unknown_targets(tmp_path):
    from elresearch.common import save_json
    from elresearch.publication_evaluate import evaluate_development
    save_json(tmp_path/'run.json',{'model':{'encoder':'unused'}})
    settings={'permitted_target_development_ids':['allowed']}
    for row in [{'id':'allowed','partition':'test'},{'id':'unknown','partition':'target_development'}]:
        with pytest.raises(PermissionError,match='predeclared'):
            evaluate_development(settings,tmp_path,[row])
