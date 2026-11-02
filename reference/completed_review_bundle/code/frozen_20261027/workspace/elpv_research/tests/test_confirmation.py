"""Synthetic-file fixtures only: these tests never open real confirmation images."""
import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from elresearch.common import digest,read_json,save_json,sha256
from elresearch import confirmation_registry as gate
from elresearch.confirmation_data import native_annotation


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(gate,'EVALUATOR_MODULES',[])
    monkeypatch.setattr(gate,'verify_source_implementation',lambda *args:[])
    # Benchmark coverage/math is tested separately. These isolated fixtures focus
    # on confirmation identity/history guards without running97 real systems.
    monkeypatch.setattr(gate,'verified_inference_efficiency',lambda:{'fixture_only':True})
    monkeypatch.setattr(gate,'ALLOWED_COHORTS',{'fixture':('test',2)})
    root=tmp_path/'artifacts/publication_20261017';root.mkdir(parents=True)
    folder=root/'final_confirmation_v1'
    source=[{'id':f's{i}','dataset':'fixture','split_group':f'source_group{i}',
             'd4_pixel_sha256':f'source_orbit{i}'} for i in range(4)]
    source_manifest=tmp_path/'source_manifest.json';save_json(source_manifest,{'records':source})
    index={'source_manifests':[gate.bind_file(source_manifest)],'records':source}
    index['signature']=digest(index);save_json(tmp_path/'source_index.json',index)
    target=[]
    for i in range(2):
        path=tmp_path/f'image{i}.png';Image.fromarray(np.full((18,37),i*255,np.uint8)).save(path)
        target.append({'id':f't{i}','dataset':'fixture','partition':'test','target':i,
                       'image_path':str(path),'image_sha256':sha256(path),
                       'split_group':f'target_group{i}','d4_pixel_sha256':f'target_orbit{i}'})
    manifest=tmp_path/'data/fixture/manifest.json';save_json(manifest,{'records':target})
    evidence={}
    for name in gate.REQUIRED_DEVELOPMENT:
        study=root/name;run=study/'runs/fit';run.mkdir(parents=True)
        checkpoint=run/('model.joblib' if name in {'classical','representation_scale'} else 'best.pt')
        checkpoint.write_bytes(b'fixture-only fitted parameters')
        status={'state':'complete','completed_runs':1,'source_calibration_complete':1}
        protocol=({'planned_real_domain_runs':1,'calibration':{'planned_runs':0}} if name=='dense'
                  else {'development_logical_runs':1,'source_calibration_trials':1} if name=='classical'
                  else {'planned_runs':1})
        protocol.update(input_size=32,batch_size=1)
        save_json(study/'protocol.json',protocol);save_json(study/'status.json',status)
        save_json(run/'run.json',{'seed':11,'training_budget_fraction':1.,'training_ids':['s0','s1'],
                 'validation_ids':['s2','s3'],'source_calibrated_margin_threshold':.5,'source_temperature':1.,
                 'source_reload_verified':True,'source_reload_maximum_absolute_error':0.,'checkpoint_sha256':sha256(checkpoint)})
        evidence[name]={'status':gate.bind_file(study/'status.json'),'protocol':gate.bind_file(study/'protocol.json'),
                        'expected_runs':1,'completed_run_glob':'runs/*/run.json'}
    model=gate.bind_model('classical','fixed',11,root/'classical/runs/fit',root/'classical/protocol.json')
    evaluator=tmp_path/'evaluator.py';evaluator.write_text('# frozen test-fixture scorer\n')
    plan={'primary_comparisons':['fixture comparison'],'primary_endpoints':['AUROC'],
          'selection_rationale':'source-only fixture','uncertainty':'fixture only','limits':['not scientific evidence'],
          'inference_device':'cpu'}
    return {'folder':folder,'models':[model],'cohorts':[gate.cohort_binding('fixture')],
            'development_evidence':evidence,'analysis_plan':plan,'evaluator_files':[evaluator],
            'source_identity_index':gate.bind_file(tmp_path/'source_index.json')}


def test_confirmation_requires_complete_development_and_actual_checkpoints(setup):
    d=copy.deepcopy(setup);d['development_evidence'].pop('representation_scale')
    with pytest.raises(RuntimeError,match='representation/scale'):gate.freeze_registry(**d)
    source=gate.check_file(setup['development_evidence']['image']['status'])
    save_json(source,{'state':'running','completed_runs':0})
    setup['development_evidence']['image']['status']=gate.bind_file(source)
    with pytest.raises(RuntimeError,match='incomplete'):gate.freeze_registry(**setup)


def test_missing_formal_efficiency_cannot_be_replaced_by_implementation_smokes(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='efficiency must be measured'):gate.verified_inference_efficiency()


def test_frozen_threshold_cannot_differ_from_verified_source(setup):
    setup['models'][0]['source_threshold']=.1
    with pytest.raises(RuntimeError,match='threshold differs'):gate.freeze_registry(**setup)


def test_duplicate_deterministic_fits_are_not_three_independent_seeds(setup):
    model=setup['models'][0]
    setup['models']=[{**copy.deepcopy(model),'seed':seed,'id':f'fixed_seed{seed}'} for seed in [11,22,33]]
    with pytest.raises(ValueError,match='counted once'):gate.freeze_registry(**setup)


def test_stochastic_family_must_retain_all_three_registered_seeds(setup):
    setup['models'][0]['kind']='image'
    with pytest.raises(ValueError,match='11/22/33'):gate.freeze_registry(**setup)


@pytest.mark.parametrize('leak',['group','orbit'])
def test_confirmation_rejects_capture_or_d4_leakage(setup,leak):
    path=gate.check_file(setup['cohorts'][0]['manifest']);d=read_json(path)
    d['records'][0]['split_group' if leak=='group' else 'd4_pixel_sha256']='source_group0' if leak=='group' else 'source_orbit0'
    save_json(path,d);setup['cohorts']=[gate.cohort_binding('fixture')]
    with pytest.raises(PermissionError,match='overlap|leakage'):gate.freeze_registry(**setup)


def test_registry_cannot_be_replaced_or_reset_in_another_folder(setup):
    registry=gate.freeze_registry(**setup)
    folder=setup['folder']
    with pytest.raises(PermissionError):gate.authorize_rows(folder,'unknown','fixture')
    assert not (folder/'observation_opened.json').exists()
    verified,model,cohort,rows=gate.authorize_rows(folder,'fixed_seed11','fixture')
    assert verified['registry_id']==registry['registry_id'] and len(rows)==2
    assert (folder/'observation_opened.json').exists()
    with pytest.raises(RuntimeError,match='already frozen'):gate.freeze_registry(**setup)
    with pytest.raises(PermissionError,match='single canonical'):
        gate.freeze_registry(**{**setup,'folder':folder.parent/'new_confirmation'})
    # Exact continuation is allowed, while changes to any evaluator file are not.
    gate.authorize_rows(folder,'fixed_seed11','fixture')
    setup['evaluator_files'][0].write_text('# changed scorer\n')
    with pytest.raises(RuntimeError,match='Frozen file changed'):gate.authorize_rows(folder,'fixed_seed11','fixture')


def test_native_ucf_ignore_policy_and_unknown_pvel_negatives(tmp_path):
    raw=np.array([[0,128,1,2,3,4]],np.uint8)
    path=tmp_path/'mask.png';Image.fromarray(raw).save(path)
    row={'id':'fixture','dataset':'ucf','partition':'external','target':1,
         'mask_path':str(path),'mask_sha256':sha256(path)}
    annotation=native_annotation(row)
    np.testing.assert_array_equal(annotation['mask'],[[0,255,1,2,2,2]])
    with pytest.raises(ValueError,match='disagree'):native_annotation({**row,'target':0})
    for target,role in [(None,'other_defect_unknown'),(0,'annotated_external_candidate')]:
        with pytest.raises((ValueError,PermissionError)):
            native_annotation({'dataset':'pvelad','partition':'external','target':target,'role':role})
    assert native_annotation({'dataset':'pvelad','partition':'external','target':0,'role':'author_normal_negative'})['kind']=='author_normal'


def test_image_only_pvel_cohort_cannot_silently_accept_unknown_negatives(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    name='pvelad_restricted_confirmation_v2'
    monkeypatch.setattr(gate,'ALLOWED_COHORTS',{name:('external',2)})
    rows=[{'id':'normal','dataset':'pvelad','partition':'external','target':0,'role':'other_defect_unknown'},
          {'id':'positive','dataset':'pvelad','partition':'external','target':1,'role':'known_annotated_crack_positive'}]
    path=tmp_path/'data'/name/'manifest.json';save_json(path,{'records':rows})
    with pytest.raises(PermissionError,match='author-normal'):gate.cohort_binding(name)


def test_streamed_confirmation_is_label_blind_and_completed_job_not_refitted(setup,monkeypatch):
    from elresearch import confirmation_evaluate as module
    gate.freeze_registry(**setup);calls=[]
    class Scorer:
        def __init__(self,binding,device):self.size=32;self.batch_size=1;calls.append('constructed')
        def __call__(self,batch):
            assert not {'y','target','mask','targets'}&set(batch)
            calls.append('scored')
            return {'image_logits':np.array([float(g.mean()/255) for g in batch['gray_images']])}
    monkeypatch.setattr(module,'FrozenScorer',Scorer)
    result=module.evaluate(setup['folder'],'fixed_seed11','fixture','cpu')
    assert result['metrics']['auc']==1 and result['metrics']['accuracy']==1
    assert result['threshold_fitted_on_target'] is False
    assert result['development_exposure']['target_dataset_used_for_fitting_or_checkpoint_validation'] is True
    assert result['development_exposure']['target_dataset_used_for_hyperparameter_selection'] is False
    assert calls==['constructed','scored','scored']
    assert module.evaluate(setup['folder'],'fixed_seed11','fixture','cpu')==result
    assert calls==['constructed','scored','scored']


def test_native_stream_accumulator_matches_full_pixel_metrics(tmp_path,monkeypatch):
    from elresearch import confirmation_evaluate as module
    from elresearch.localization_metrics import pooled_native_metrics
    positive=np.array([[0,1,1],[0,0,255]],np.uint8);negative=np.zeros((2,3),np.uint8)
    maps=[np.array([[.2,.8,.7],[.1,.6,.9]],np.float32),np.full((2,3),.1,np.float32)]
    masks={'P':positive,'N':negative}
    monkeypatch.setattr(module,'native_annotation',lambda row:{'kind':'pixel_mask','mask':masks[row['id']],'normal_structure':None})
    acc=module.NativeAccumulator(tmp_path/'pixel_cache',fixed_ids=['P'])
    for name,values in zip(['P','N'],maps):acc.consume({'id':name,'split_group':name},0.,values)
    result=acc.finish();expected=pooled_native_metrics([{'mask':m,'score':p} for m,p in zip([positive,negative],maps)])
    assert result['pooled_native_pixel_AP']==expected['pooled_pixel_average_precision']
    assert result['mean_positive_image_dice']==expected['mean_positive_image_dice']
    assert result['mean_negative_image_pixel_FPR']==0
    assert len(list((tmp_path/'fixed_native_examples').glob('*.npz')))==1


def test_frozen_image_scorer_keeps_source_cached_head_prediction_path(tmp_path,monkeypatch):
    from transformers import DINOv3ViTModel,DINOv3ViTConfig
    from elresearch import confirmation_predict as module
    from elresearch.publication_models import PublicationClassifier
    from elresearch.image_classification import trainable_state
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    def tiny():
        return DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32,num_hidden_layers=4,num_attention_heads=4,
                                            intermediate_size=64,patch_size=16,num_register_tokens=4))
    initial=copy.deepcopy(tiny().state_dict());identity={'fixture':'not research encoder'}
    def backbone(name,size,device):
        model=tiny();model.load_state_dict(initial);return model.to(device),32,identity
    model=PublicationClassifier(backbone('',32,'cpu')[0],32,'dinov3_small','frozen',11).eval()
    checkpoint=tmp_path/'best.pt';torch.save(trainable_state(model),checkpoint)
    save_json(tmp_path/'run.json',{'model':{'encoder':'dinov3_small','adaptation':'frozen'},'seed':11,
                                  'encoder':identity,'cached_augmented_features':True})
    save_json(tmp_path/'protocol.json',{'input_size':32,'batch_size':2})
    binding={'kind':'image','run':gate.bind_file(tmp_path/'run.json'),'source_protocol':gate.bind_file(tmp_path/'protocol.json'),
             'checkpoint':gate.bind_file(checkpoint)}
    monkeypatch.setattr(module,'build_backbone',backbone)
    monkeypatch.setattr(module,'_check_source_code',lambda *args:None)
    scorer=module.FrozenScorer(binding,'cpu');images=torch.randn(2,3,32,32)
    with torch.no_grad():expected=model(images)
    np.testing.assert_allclose(scorer({'images':images})['image_logits'].numpy(),expected.numpy(),atol=1e-6)
    with pytest.raises(PermissionError):scorer({'images':images,'target':torch.tensor([0,1])})


def test_upstream_hyperparameter_label_exposure_is_not_hidden(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    parent=tmp_path/'artifacts/publication_20261017/calibration_v1/protocol.json'
    save_json(parent,{'source_training_ids':['ucf/train0'],'source_validation_ids':['ucf/val0']})
    child=tmp_path/'scdd_protocol.json';save_json(child,{'calibration_protocol_sha256':sha256(parent)})
    ids,lineage=gate.selection_exposure(child)
    assert ids==['ucf/train0','ucf/val0'] and lineage==[gate.bind_file(parent)]
    save_json(parent,{'source_training_ids':[],'source_validation_ids':[]})
    with pytest.raises(RuntimeError,match='calibration changed'):gate.selection_exposure(child)


def test_frozen_representation_scorer_uses_registered_pooling_and_size(tmp_path,monkeypatch):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from elresearch import confirmation_predict as module
    from elresearch.representation_features import normalized_representation
    cls=np.array([[[1.,3.,2.]],[[3.,1.,2.]]],np.float32);mean=cls*2+1
    features=normalized_representation(cls,mean,'concat')[:,0]
    model=LogisticRegression().fit(features,[0,1]);path=tmp_path/'model.joblib';joblib.dump(model,path)
    run=tmp_path/'run.json';protocol=tmp_path/'protocol.json'
    save_json(run,{'dimensions':6,'feature_identity':{'kind':'frozen_representation_v1','input_size':224,
        'encoder':{'name':'fixture'},'pooling':'concat'}})
    save_json(protocol,{'feature_batch_size':2})
    def binding(path):return {'path':str(path),'sha256':sha256(path)}
    monkeypatch.setattr(module,'check_file',lambda b:Path(b['path']))
    monkeypatch.setattr(module,'_check_source_code',lambda *a:None)
    class Fixture:
        def __init__(self,identity,size,device):assert size==224 and identity['name']=='fixture'
        def encode(self,images,views):
            assert views==[0] and all(im.mode=='L' for im in images)
            return cls,mean
    monkeypatch.setattr(module,'FrozenRepresentationEncoder',Fixture)
    scorer=module.FrozenScorer({'kind':'classical','checkpoint':binding(path),'run':binding(run),
        'source_protocol':binding(protocol)},'cpu')
    result=scorer({'gray_images':[np.zeros((20,30),np.uint8)]*2})['image_logits']
    np.testing.assert_array_equal(result,model.decision_function(features.astype(np.float64)))
    assert scorer.size==224 and scorer.batch_size==2
    with pytest.raises(PermissionError):scorer({'gray_images':[],'mask':np.zeros((20,30))})


def test_ssl_unlabelled_source_orbits_are_included_in_confirmation_leak_guard(setup):
    root=gate.ROOT/'artifacts/publication_20261017'
    folder=root/'classical/runs/fit';run=read_json(folder/'run.json')
    run['unlabelled_source_ids']=['u0'];save_json(folder/'run.json',run)
    index_path=gate.check_file(setup['source_identity_index']);index=read_json(index_path)
    row={'id':'u0','dataset':'fixture','split_group':'ul_group','d4_pixel_sha256':'ul_orbit'}
    source_manifest=gate.check_file(index['source_manifests'][0]);source=read_json(source_manifest)
    source['records'].append(row);save_json(source_manifest,source)
    index['records'].append(row);index['source_manifests']=[gate.bind_file(source_manifest)]
    index['signature']=digest({k:v for k,v in index.items() if k!='signature'});save_json(index_path,index)
    setup['source_identity_index']=gate.bind_file(index_path)
    setup['models']=[gate.bind_model('classical','fixed',11,folder,root/'classical/protocol.json')]
    assert setup['models'][0]['source_unlabelled_ids']==['u0']
    target=gate.check_file(setup['cohorts'][0]['manifest']);d=read_json(target)
    d['records'][0]['d4_pixel_sha256']='ul_orbit';save_json(target,d)
    setup['cohorts']=[gate.cohort_binding('fixture')]
    with pytest.raises(PermissionError,match='duplicate leakage'):gate.freeze_registry(**setup)


def test_published_views_bind_their_own_source_threshold_and_complete_archive_dependencies(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(gate,'verify_source_implementation',lambda *args:[])
    folder=tmp_path/'fit';folder.mkdir();(folder/'best.pt').write_bytes(b'fixture compressed state')
    backbone=tmp_path/'backbone.pt';backbone.write_bytes(b'exact fixture backbone')
    heads=tmp_path/'basis.pt';heads.write_bytes(b'exact fixture heads')
    proof={'fresh_model':True,'logit_max_abs_error':0.,'sampled_pixel_probability_max_abs_error':0.}
    views={name:{'source_logit_threshold':value,'source_temperature':1.,'checkpoint_reload':proof}
           for name,value in [('primary',.1),('ensemble',.2)]}
    run={'seed':11,'training_budget_fraction':.2,'training_ids':['l0'],'validation_ids':['v0'],
         'unlabelled_source_ids':['u0'],'checkpoint_sha256':sha256(folder/'best.pt'),
         'model':{'architecture':'published_psmt_deeplab','semi':True,
             'archive_basis':{**gate.bind_file(heads),'backbone_sha256':sha256(backbone)},
             'backbone_sha256':sha256(backbone)},'encoder':{'backbone':gate.bind_file(backbone)},
         'inference_views':views,**views['primary']}
    save_json(folder/'run.json',run);protocol=tmp_path/'protocol.json';save_json(protocol,{})
    primary=gate.bind_model('dense','published_primary',11,folder,protocol)
    ensemble=gate.bind_model('dense','published_ensemble',11,folder,protocol,'ensemble')
    assert primary['source_threshold']==.1 and ensemble['source_threshold']==.2
    assert ensemble['source_view']=='ensemble' and ensemble['source_unlabelled_ids']==['u0']
    assert ensemble['frozen_model_dependencies']==[gate.bind_file(heads),gate.bind_file(backbone)]
    with pytest.raises(ValueError):gate.bind_model('dense','published',11,folder,protocol,'best_on_target')
    run['inference_views']['ensemble']['checkpoint_reload']={**proof,'logit_max_abs_error':.01}
    save_json(folder/'run.json',run)
    with pytest.raises(RuntimeError,match='fresh-model'):gate.bind_model('dense','published',11,folder,protocol,'ensemble')
    run['implementation_smoke']=True;save_json(folder/'run.json',run)
    with pytest.raises(PermissionError,match='smoke'):gate.bind_model('dense','published',11,folder,protocol)


def test_frozen_scorer_routes_full_published_archive_and_predeclared_teacher_view(tmp_path,monkeypatch):
    from elresearch import confirmation_predict as module
    run=tmp_path/'run.json';protocol=tmp_path/'protocol.json';checkpoint=tmp_path/'best.pt'
    save_json(run,{'model':{'architecture':'published_psmt_deeplab'}})
    save_json(protocol,{'input_size':336,'batch_size':2});checkpoint.write_bytes(b'fixture')
    def binding(path):return {'path':str(path),'sha256':sha256(path)}
    monkeypatch.setattr(module,'check_file',lambda b:Path(b['path']))
    monkeypatch.setattr(module,'_check_source_code',lambda *a:None)
    class Fixture(torch.nn.Module):
        def forward(self,images,valid,geometries,groups):
            return {'image':images[:,0,0,0],'segmentation':images}
    calls=[]
    def loader(run,path,device,view):calls.append((path,str(device),view));return Fixture().eval()
    monkeypatch.setattr(module,'load_published_inference',loader)
    scorer=module.FrozenScorer({'kind':'dense','source_view':'ensemble','run':binding(run),
        'source_protocol':binding(protocol),'checkpoint':binding(checkpoint)},'cpu')
    images=torch.randn(2,3,5,5)
    output=scorer({'images':images,'valid':torch.ones(2,5,5,dtype=torch.bool),
                   'geometries':[{},{}],'groups':['a','b']})
    assert calls==[(checkpoint,'cpu','ensemble')]
    torch.testing.assert_close(output['crack_probability'],images.softmax(1)[:,1])
    with pytest.raises(PermissionError):scorer({'images':images,'target':torch.zeros(2)})


def test_primary_published_scorer_does_not_keep_inactive_teacher_in_device_memory(tmp_path,monkeypatch):
    from elresearch import confirmation_predict as module
    run=tmp_path/'run.json';protocol=tmp_path/'protocol.json';checkpoint=tmp_path/'best.pt'
    save_json(run,{'model':{'architecture':'published_psmt_deeplab','semi':True}})
    save_json(protocol,{'input_size':336,'batch_size':2});checkpoint.write_bytes(b'fixture')
    def binding(path):return {'path':str(path),'sha256':sha256(path)}
    monkeypatch.setattr(module,'check_file',lambda b:Path(b['path']))
    monkeypatch.setattr(module,'_check_source_code',lambda *a:None)
    class Twin(torch.nn.Module):
        def __init__(self):
            super().__init__();self.first=torch.nn.Linear(3,1);self.second=torch.nn.Linear(3,1)
    source=Twin().eval();expected=copy.deepcopy(source.first.state_dict())
    monkeypatch.setattr(module,'load_published_inference',lambda *a:source)
    scorer=module.FrozenScorer({'kind':'dense','source_view':'primary','run':binding(run),
        'source_protocol':binding(protocol),'checkpoint':binding(checkpoint)},'cpu')
    assert not hasattr(scorer.model,'second')
    assert sum(p.numel() for p in scorer.model.parameters())==4
    for key,value in scorer.model.first.state_dict().items():torch.testing.assert_close(value,expected[key])
