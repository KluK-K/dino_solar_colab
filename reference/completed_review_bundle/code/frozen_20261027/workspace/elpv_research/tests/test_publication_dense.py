"""Geometry, reference leakage and fresh-reload tests; fixtures are not evidence."""
import copy
import numpy as np
import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from elresearch.encoders import geometry
from elresearch.publication_models import d4_view
from elresearch.publication_dense_data import rotate_geometry, mask_target, prepare_dense_rows
from elresearch.publication_dense_models import make_dense_dino, PublicationDeepLabPlus
from elresearch.publication_normal_bank import OrientationNormalBank
from elresearch.image_classification import trainable_state


def tiny():
    return DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=64, patch_size=16, num_register_tokens=4))


def test_synchronized_d4_geometry_with_asymmetric_odd_padding():
    g = geometry(401, 207, 64)
    image = torch.zeros(64, 64, dtype=torch.bool)
    image[g['top']:g['top']+g['h'], g['left']:g['left']+g['w']] = True
    for code in range(8):
        rotated = rotate_geometry(g, code)
        expected = torch.zeros_like(image)
        expected[rotated['top']:rotated['top']+rotated['h'], rotated['left']:rotated['left']+rotated['w']] = True
        assert torch.equal(d4_view(image, code), expected)
        assert (rotated['width'], rotated['height']) == ((207,401) if code%2 else (401,207))
    mask = np.zeros((207,401), np.uint8); mask[101,201] = 1
    assert (mask_target(mask,g) == 1).any()


def test_dense_development_and_reference_guards_run_before_file_access():
    with pytest.raises(PermissionError):
        prepare_dense_rows([{'partition':'test'}],336)
    with pytest.raises(PermissionError):
        OrientationNormalBank.build([{'partition':'validation'}],None,336,{},'cpu')


def test_reference_excludes_query_family_and_matches_orientation():
    def item(group, vector, other):
        views = [{'cls':torch.tensor([1.,0.]), 'patch':torch.tensor([vector]), 'coords':torch.tensor([[.5,.5]])} for _ in range(8)]
        views[1] = {**views[1], 'patch':torch.tensor([other])}
        return {'group':group,'views':views}
    bank = OrientationNormalBank([item('A',[0.,1.],[1.,0.]),item('B',[1.,0.],[0.,1.])],{})
    feature = {'cls':np.array([1.,0.]), 'patch':np.array([[[1.,0.]]]), 'coords':np.array([[[.5,.5]]])}
    out = bank.residual(feature,('A',0),geometry_weight=0)
    np.testing.assert_allclose(out.numpy()[0,0,2:4],[0,0])
    rotated = bank.residual(feature,('A',1),geometry_weight=0)
    np.testing.assert_allclose(rotated.numpy()[0,0,2:4],[1,-1])
    with pytest.raises(ValueError): bank.residual(feature,'A')
    with pytest.raises(ValueError): OrientationNormalBank([item('A',[1.,0.],[1.,0.])],{})


def test_identical_dino_heads_across_adaptation_and_frozen_reference_slots():
    initial = copy.deepcopy(tiny().state_dict()); states = []
    for adaptation in ['frozen','lora','full']:
        backbone=tiny();backbone.load_state_dict(initial)
        model=make_dense_dino(backbone,adaptation,11)
        states.append({k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')})
    assert all(torch.equal(value,states[1][key]) and torch.equal(value,states[2][key]) for key,value in states[0].items())
    with pytest.raises(ValueError): make_dense_dino(tiny(),'lora',11,bank=object())


def test_frozen_reference_recipe_identity_and_payload_hash(tmp_path,monkeypatch):
    from elresearch import publication_normal_bank as module
    from elresearch.image_classification import atomic_torch_save
    monkeypatch.setattr(module,'ROOT',tmp_path)
    items=[]
    for name in ['A','B']:
        item={'id':name,'group':name,'cache_key':name,'image_sha256':'image'+name,'mask_sha256':'mask'+name,
              'views':[{'cls':torch.ones(2),'patch':torch.ones(12,2),'coords':torch.ones(12,2)} for _ in range(8)]}
        items.append(item)
        path=tmp_path/'cache/publication_normal_d4'/f'{name}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        atomic_torch_save(item,path)
    folder=tmp_path/'run';folder.mkdir()
    module.OrientationNormalBank(items,{}).write_recipe(folder)
    assert len(module.OrientationNormalBank.from_recipe(folder/'normal_bank_recipe.json',['A','B']).items)==2
    with pytest.raises(PermissionError): module.OrientationNormalBank.from_recipe(folder/'normal_bank_recipe.json',['A'])
    atomic_torch_save({'corrupt':True},tmp_path/'cache/publication_normal_d4/A.pt')
    with pytest.raises(RuntimeError): module.OrientationNormalBank.from_recipe(folder/'normal_bank_recipe.json',['A','B'])


def test_dense_evaluator_rejects_sealed_and_unregistered_before_access(tmp_path):
    from elresearch.publication_dense_evaluate import evaluate_development
    for row in [{'id':'allowed','partition':'test'}, {'id':'unknown','partition':'target_development'}]:
        with pytest.raises(PermissionError):
            evaluate_development({'permitted_target_development_ids':['allowed']},tmp_path,[row])


def test_full_cnn_skip_decoder_and_joint_gradient_contract():
    from torchvision.models import resnet50
    model=PublicationDeepLabPlus(resnet50(weights=None,replace_stride_with_dilation=[False,False,True]),11).train()
    before=model.encoder.bn1.running_mean.clone()
    image=torch.randn(2,3,32,32);valid=torch.ones(2,32,32,dtype=torch.bool)
    output=model(image,valid)
    assert output['segmentation'].shape==(2,3,32,32)
    from elresearch.full_models import dense_loss
    target=torch.zeros(2,32,32,dtype=torch.long);target[:,16,10:18]=1
    dense_loss(output,target,torch.ones(2),torch.ones(3)).backward()
    assert model.encoder.layer4[-1].conv3.weight.grad.abs().sum()>0
    assert model.segment.weight.grad.abs().sum()>0 and model.image_head.linear.weight.grad.abs().sum()>0
    assert torch.equal(before,model.encoder.bn1.running_mean)


@pytest.mark.parametrize('adaptation',['frozen','lora','full'])
def test_dense_training_fresh_checkpoint_reload(tmp_path,monkeypatch,adaptation):
    from elresearch import publication_dense_train as trainer
    initial=copy.deepcopy(tiny().state_dict())
    def build(spec,seed,size,device,training_rows=None,bank=None):
        backbone=tiny();backbone.load_state_dict(initial)
        return make_dense_dino(backbone,spec['adaptation'],seed).to(device),{'fixture':True},None
    def examples(rows,size):
        result=[]
        for row in rows:
            target=np.zeros((size,size),np.uint8)
            if row['target']: target[16,10:20]=1
            result.append({'row':row,'image':torch.randn(3,size,size)*0+row['target'],
                           'geometry':geometry(size,size,size),'target':torch.tensor(target).long(),
                           'valid':torch.ones(size,size,dtype=torch.bool),'native_mask':target,
                           'native_structure':None,'selection_indices':np.arange(size*size)})
        return result
    hashes=trainer.scientific_hashes()
    monkeypatch.setattr(trainer,'scientific_hashes',lambda:hashes)
    monkeypatch.setattr(trainer,'build_system',build)
    monkeypatch.setattr(trainer,'prepare_dense_rows',examples)
    rows=[{'id':str(i),'partition':'train' if i<4 else 'validation','split_group':str(i),
           'pixel_sha256':str(i),'image_sha256':str(i),'target':i%2,'crack_area_fraction':.01*(i%2)} for i in range(8)]
    settings={'study_id':'test-only','scientific_hashes':hashes,'device':'cpu','input_size':32,
              'batch_size':2,'updates_per_round':1,'maximum_rounds':1,'patience':1,'weight_decay':1e-4,
              'selection':'fixture','target_source_recall':.95}
    spec={'name':'fixture','architecture':'fixture','adaptation':adaptation,'encoder_lr':1e-4,'head_lr':1e-3}
    result=trainer.train(settings,spec,rows[:4],rows[4:],11,1.,tmp_path)
    assert result['checkpoint_reload']['fresh_model']
    assert result['checkpoint_reload']['logit_max_abs_error']<2e-5
    assert result['training_images']==4 and result['total_pixel_annotated_images_including_validation']==8
    assert result['first_gradient_check']['encoder_has_gradient']==(adaptation!='frozen')
    assert trainer.train(settings,spec,rows[:4],rows[4:],11,1.,tmp_path)==result
