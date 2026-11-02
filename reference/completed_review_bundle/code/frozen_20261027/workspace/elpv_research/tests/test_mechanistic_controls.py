from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from elresearch.mechanistic_models import LocalPatchDINO,GlobalNormalDistance,global_normal_distance
from elresearch.mechanistic_train import mechanistic_loss
from elresearch.encoders import patch_targets
from elresearch.adapt import inject_lora
from elresearch.common import ROOT,read_json,save_json,sha256


class FixtureEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=8,patch_size=16,num_register_tokens=4)
        self.linear=nn.Linear(3,8)
    def forward(self,pixel_values):
        patches=F.avg_pool2d(pixel_values,16,16).flatten(2).transpose(1,2)
        patches=self.linear(patches);cls=patches.mean(1,keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.cat([cls,torch.zeros_like(cls).expand(-1,4,-1),patches],1))


def test_local_patch_supervision_preserves_thin_cracks_and_freezes_encoder():
    encoder=FixtureEncoder();model=LocalPatchDINO(encoder,11).train()
    images=torch.randn(2,3,64,64);valid=torch.ones(2,64,64,dtype=torch.bool)
    output=model(images,valid)
    target=np.zeros((64,64),np.uint8);target[1,1]=1
    g={'width':64,'height':64,'w':64,'h':64,'top':0,'left':0,'size':64}
    patch=patch_targets(target,g,16)
    assert patch[0,0]==1 and (patch==1).sum()==1
    batch={'patch_target':torch.tensor(np.stack([patch,patch])),'target':torch.tensor(np.stack([target,target])),
        'y':torch.tensor([1.,1.])}
    loss=mechanistic_loss(output,batch,torch.ones(3));loss.backward()
    assert torch.isfinite(loss) and output['segmentation'].shape==(2,3,64,64)
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.patch_head.parameters())
    assert not encoder.training


def normal_item(group,vector):
    return {'group':group,'views':[{'patch':torch.tensor([vector],dtype=torch.float16)} for _ in range(8)]}


def test_normal_distance_excludes_entire_query_family_and_chunking_is_exact():
    query=torch.tensor([[[1.,0.],[0.,1.]]])
    items=[normal_item('own',[1.,0.]),normal_item('other',[0.,1.])]
    value=global_normal_distance(query,items,('own',0),chunk=1)
    torch.testing.assert_close(value,torch.tensor([[1.,0.]]))
    with pytest.raises(ValueError,match='eligible'):
        global_normal_distance(query,[items[0]],('own',0))
    with pytest.raises(ValueError,match='configuration'):
        global_normal_distance(query,items,('new',8))


def test_distance_calibration_is_a_saved_source_cutoff_not_encoder_updates():
    bank=SimpleNamespace(items=[normal_item('a',[1.]*8),normal_item('b',[-1.]*8)])
    model=GlobalNormalDistance(FixtureEncoder(),bank,1.).train()
    assert not model.training and not any(p.requires_grad for p in model.encoder.parameters())
    model.load_calibration_state({'pixel_cutoff':torch.tensor(.2)})
    assert float(model.pixel_cutoff)==pytest.approx(.2)
    with pytest.raises(ValueError,match='Invalid'):
        model.load_calibration_state({'pixel_cutoff':torch.tensor(float('nan'))})
    with pytest.raises(ValueError,match='Unexpected'):
        model.load_calibration_state({'encoder.weight':torch.zeros(1)})


def test_stale_pre_adaptation_normal_bank_is_rejected(monkeypatch):
    from elresearch import mechanistic_models as module
    class Encoder:
        def __init__(self,*args,**kwargs):self.model=FixtureEncoder();self.identity={'weights_sha256':'new'}
    monkeypatch.setattr(module,'FrozenEncoder',Encoder)
    stale=SimpleNamespace(encoder_identity={'weights_sha256':'old'})
    with pytest.raises(RuntimeError,match='pre-adaptation'):
        module.build_mechanistic_system({'architecture':'dinov3_normal_distance','reference':True},11,64,'cpu',bank=stale)


def test_post_lora_merge_export_and_reuse_detect_changed_weights(tmp_path,monkeypatch):
    from PIL import Image
    from transformers import AutoModel,DINOv3ViTConfig
    from elresearch import postadapt_encoder as module
    configuration=DINOv3ViTConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=12,
        num_attention_heads=4,image_size=32,patch_size=16,num_register_tokens=4)
    class TinyFrozen:
        def __init__(self,name,size,device,model_path=None):
            if model_path:
                self.model=AutoModel.from_pretrained(model_path,local_files_only=True)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(7);self.model=AutoModel.from_config(configuration)
            self.model.eval().requires_grad_(False).to(device)
            self.identity={'fixture':'synthetic HF architecture, no real model result'}
    monkeypatch.setattr(module,'FrozenEncoder',TinyFrozen)
    monkeypatch.setattr(module,'load_image',lambda row:Image.new('L',(32,32),80))
    (tmp_path/'elresearch').mkdir()
    (tmp_path/'elresearch/adapt.py').write_bytes((ROOT/'elresearch/adapt.py').read_bytes())
    monkeypatch.setattr(module,'ROOT',tmp_path)
    encoder=TinyFrozen('fixture',32,'cpu');inject_lora(encoder.model,rank=8,blocks=4)
    with torch.no_grad():
        for name,p in encoder.model.named_parameters():
            if name.endswith('.b'):p.normal_(0,.01)
    parent=tmp_path/'parent';parent.mkdir()
    torch.save({'encoder.'+name:p.detach() for name,p in encoder.model.named_parameters() if p.requires_grad},parent/'best.pt')
    save_json(parent/'run.json',{'model':{'adaptation':'lora','architecture':'dinov3_multidepth','reference':False},
        'seed':11,'encoder':encoder.identity,'checkpoint_sha256':sha256(parent/'best.pt'),
        'training_ids':['train0'],'validation_ids':['val0']})
    destination=tmp_path/'export';rows=[{'id':'train0'}]
    info=module.export_source_encoder(parent,rows,32,'cpu',destination)
    assert info['adapter_tensors']==16 and info['merge_max_abs_error']<3e-5
    assert module.export_source_encoder(parent,rows,32,'cpu',destination)==info
    with (destination/'model.safetensors').open('ab') as f:f.write(b'tamper')
    with pytest.raises(RuntimeError,match='changed'):
        module.export_source_encoder(parent,rows,32,'cpu',destination)
    with pytest.raises(PermissionError,match='same budgeted'):
        module.export_source_encoder(parent,[{'id':'other'}],32,'cpu',tmp_path/'wrong')


def test_mechanistic_plan_retains_local_baselines_and_both_fresh_postadapt_heads():
    from run_mechanistic_controls import registered_design
    settings=registered_design()
    assert settings['planned_local_patch_runs']==36
    assert settings['planned_distance_runs']==23
    assert settings['planned_postadapt_runs']==66 and settings['planned_runs']==125


def test_source_patch_training_loop_fresh_checkpoint_reload(tmp_path,monkeypatch):
    from PIL import Image
    from elresearch import mechanistic_models as module
    from elresearch.mechanistic_train import train,scientific_hashes
    class Frozen:
        def __init__(self,*args,**kwargs):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7);self.model=FixtureEncoder()
            self.identity={'fixture':'source training loop only'}
    monkeypatch.setattr(module,'FrozenEncoder',Frozen)
    rows=[]
    for i in range(8):
        image=tmp_path/f'image{i}.png';mask=tmp_path/f'mask{i}.png'
        Image.fromarray(np.full((64,64),40+i*10,np.uint8)).save(image)
        annotation=np.zeros((64,64),np.uint8)
        if i%2:annotation[20,10:50]=1
        annotation[40:50,20:30]=2
        Image.fromarray(annotation).save(mask)
        rows.append({'id':f'fixture{i}','dataset':'ucf','partition':'train' if i<4 else 'validation',
            'target':i%2,'crack_area_fraction':float((annotation==1).mean()),'split_group':f'group{i}',
            'image_path':str(image),'mask_path':str(mask),'image_sha256':sha256(image),'mask_sha256':sha256(mask),
            'pixel_sha256':str(i),'d4_pixel_sha256':str(i)})
    settings={'scientific_hashes':scientific_hashes(),'study_id':'synthetic-loop-fixture','device':'cpu',
        'input_size':64,'batch_size':2,'updates_per_round':1,'maximum_rounds':1,'patience':1,
        'weight_decay':1e-4,'target_source_recall':.95,'selection':'fixture source pixels only'}
    spec={'name':'fixture_local','architecture':'dinov3_local_patch','adaptation':'frozen',
        'reference':False,'encoder_lr':0.,'head_lr':1e-3}
    result=train(settings,spec,rows[:4],rows[4:],11,1.,tmp_path/'fit')
    assert result['first_gradient_check']=={'encoder_has_gradient':False,'heads_have_gradient':True}
    assert result['checkpoint_reload']['fresh_model'] and result['checkpoint_reload']['logit_max_abs_error']<2e-5
    assert result['optimizer_updates']==1 and result['training_ids']==[r['id'] for r in rows[:4]]


def test_canonical_reference_payload_preserves_inference_and_demands_training_rebuild(tmp_path,monkeypatch):
    import gzip
    from elresearch import packed_normal_bank as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    items=[]
    for i in range(2):
        views=[]
        for view in range(8):
            views.append({'cls':torch.tensor([1.,i+1],dtype=torch.float16),
                'patch':torch.tensor([[i+1.,view+1.]],dtype=torch.float16),
                'coords':torch.tensor([[.5,.5]],dtype=torch.float16)})
        items.append({'id':f's{i}','group':f'g{i}','cache_key':str(i),'image_sha256':str(i),
            'mask_sha256':str(i),'views':views})
    ids=['s0','s1'];identity={'weights_sha256':'source-merged-fixture','config':{'id2label':{0:'normal',1:'crack'}}}
    hashes=[{'id':x['id'],'view_sha256':[module.view_digest(v) for v in x['views']]} for x in items]
    payload=tmp_path/'memory.pt.gz'
    with gzip.open(payload,'wb') as f:
        torch.save({'encoder_identity':identity,'training_ids':ids,
            'items':[{**x,'views':[x['views'][0]]} for x in items]},f)
    binding={'path':'memory.pt.gz','sha256':sha256(payload)}
    live=module.PackedNormalBank(items,identity,binding,ids,hashes,range(8));live.write_recipe(tmp_path)
    loaded=module.load_reference_bank(tmp_path/'normal_bank_recipe.json',ids)
    feature={'cls':np.array([1.,1.]),'patch':np.array([[[1.,1.]]]),'coords':np.array([[[.5,.5]]])}
    torch.testing.assert_close(live.residual(feature,query_group=('new',0)),loaded.residual(feature,query_group=('new',0)),atol=0,rtol=0)
    with pytest.raises(RuntimeError,match='regenerated'):
        loaded.residual(feature,query_group=('new',1))
    with pytest.raises(PermissionError,match='budget'):
        module.load_reference_bank(tmp_path/'normal_bank_recipe.json',['s0'])
    with payload.open('ab') as f:f.write(b'tamper')
    with pytest.raises(RuntimeError,match='payload changed'):
        module.load_reference_bank(tmp_path/'normal_bank_recipe.json',ids)


def test_parent_dense_development_accepts_json_roundtrip_but_rejects_changed_config(tmp_path,monkeypatch):
    from elresearch import publication_dense_evaluate as module
    monkeypatch.setattr(module,'scientific_hashes',lambda:{'fixture':'fixed'})
    identity={'config':{'id2label':{0:'normal',1:'crack'}},'weights_sha256':'fixture'}
    monkeypatch.setattr(module,'build_system',lambda *args,**kwargs:(nn.Identity(),identity,None))
    monkeypatch.setattr(module,'prepare_dense_rows',lambda rows,size:rows)
    monkeypatch.setattr(module,'native_evaluation',lambda *args,**kwargs:(np.array([-1.,1.]),{'fixture':True}))
    target=[{'id':f't{i}','partition':'target_development','target':i,'split_group':f'g{i}'} for i in range(2)]
    settings={'permitted_target_development_ids':['t0','t1'],'scientific_hashes':{'fixture':'fixed'},
        'device':'cpu','input_size':32,'batch_size':2}
    for name in ['accepted','changed']:
        folder=tmp_path/name;folder.mkdir();torch.save({},folder/'best.pt')
        save_json(folder/'run.json',{'model':{'reference':False},'seed':11,'training_ids':['s0'],
            'validation_ids':['s1'],'checkpoint_sha256':sha256(folder/'best.pt'),'encoder':identity,
            'source_logit_threshold':0.,'source_temperature':1.})
    result=module.evaluate_development(settings,tmp_path/'accepted',target)
    assert result['metrics']['auc']==1.
    identity['config']['id2label'][1]='changed'
    with pytest.raises(RuntimeError,match='identity changed'):
        module.evaluate_development(settings,tmp_path/'changed',target)
