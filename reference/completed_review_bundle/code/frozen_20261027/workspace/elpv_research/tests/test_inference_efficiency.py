from types import SimpleNamespace
import copy

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from elresearch.common import sha256
from elresearch.inference_efficiency import (source_image_rows,prepare_batch,measure_cohort,
    ComponentProfiler,pass_summary,verify_signatures)


def fixture_rows(tmp_path):
    rows=[]
    for i,(height,width) in enumerate([(32,49),(40,27),(23,31)]):
        path=tmp_path/f'image{i}.png';Image.fromarray(np.random.default_rng(i).integers(0,256,(height,width),np.uint8)).save(path)
        rows.append({'id':str(i),'dataset':'fixture','partition':'train','image_path':str(path),
                     'image_sha256':sha256(path),'split_group':str(i),'target':i%2,'mask_path':'MUST_NOT_READ'})
    return rows


class TinyImage(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Linear(4,1)
    def forward(self,images):return self.head(self.encoder(images))[:,0]


class ImageScorer:
    def __init__(self):self.model=TinyImage().eval();self.device='cpu';self.size=32;self.binding={'kind':'image'}
    def __call__(self,batch):
        assert not {'target','mask','y'}&set(batch)
        with torch.inference_mode():return {'image_logits':self.model(batch['images'])}


def test_efficiency_rows_are_annotation_blind_and_reject_sealed_identities(tmp_path):
    rows=fixture_rows(tmp_path);permitted=[r['id'] for r in rows]
    projected=source_image_rows(rows,permitted)
    assert not {'target','mask_path','y'}&set(projected[0])
    with pytest.raises(PermissionError):source_image_rows([{**rows[0],'partition':'test'}],permitted)
    with pytest.raises(PermissionError):source_image_rows(rows,permitted[:1])
    with pytest.raises(PermissionError):prepare_batch(rows,32)
    batch,stages=prepare_batch(projected,32)
    assert batch['images'].shape==(3,3,32,32) and all(v>=0 for v in stages.values())
    from elresearch.streaming_inference import image_batches
    _,production=next(image_batches(projected,32,3))
    for key in ['images','valid']:torch.testing.assert_close(batch[key],production[key])
    assert batch['geometries']==production['geometries'] and batch['ids']==production['ids']
    for first,second in zip(batch['gray_images'],production['gray_images']):np.testing.assert_array_equal(first,second)


def test_component_hooks_preserve_predictions_and_clean_up_after_exception():
    scorer=ImageScorer();batch={'images':torch.randn(2,3,32,32)};before=scorer(batch)['image_logits']
    with pytest.raises(RuntimeError,match='fixture'):
        with ComponentProfiler(scorer) as profiler:
            after=scorer(batch)['image_logits'];torch.testing.assert_close(before,after)
            assert set(profiler.elapsed)=={'feature_extraction','classification'}
            assert all(n==1 for n in profiler.calls.values())
            raise RuntimeError('fixture')
    assert not scorer.model.encoder._forward_hooks and not scorer.model.head._forward_pre_hooks
    torch.testing.assert_close(before,scorer(batch)['image_logits'])


def test_full_timing_passes_cover_partial_batches_and_actual_native_geometry(tmp_path):
    class DenseScorer(ImageScorer):
        def __call__(self,batch):
            output=super().__call__(batch)
            output['crack_probability']=batch['images'][:,0].sigmoid();return output
    rows=source_image_rows(fixture_rows(tmp_path),['0','1','2']);scorer=DenseScorer()
    settings={'batch_sizes':[1,2],'warmup_batches':1,'repetitions':2,'seed':1,'memory_sampling_interval':.01}
    path=tmp_path/'signatures.npz';result=measure_cohort(scorer,rows,settings,0.,1.,'fixture',path)
    assert len(result)==2
    for case in result:
        assert len(case['repeated_uninstrumented_passes'])==2
        for measurement in case['repeated_uninstrumented_passes']+[case['component_pass']]:
            assert measurement['images']==3 and measurement['weighted_mean_ms_per_image']>0
            assert measurement['stage_sum_seconds']['decision_and_native_reconstruction']>0
        assert len(case['prediction_equivalence'])==3 and case['memory']['samples']>0
    with np.load(path,allow_pickle=False) as f:
        assert list(f['ids'])==['0','1','2'] and len(f['native_sample_offsets'])==4
        assert len(f['native_sample_values'])==sum(min(512,np.prod(shape)) for shape in [(32,49),(40,27),(23,31)])


def test_partial_batch_throughput_uses_image_counts_not_nominal_batch_size():
    rows=[{'images':8,'seconds':.08,'stages':{'forward':.08}},{'images':1,'seconds':.03,'stages':{'forward':.03}}]
    r=pass_summary(rows,.12)
    assert r['images']==9 and r['pipeline_images_per_second']==9/.11
    assert r['observed_loop_images_per_second']==9/.12


def test_efficiency_decisions_use_exact_source_calibration_strict_tie_convention():
    from elresearch.inference_efficiency import source_decisions
    from elresearch.metrics import classification
    logits=np.array([-.5,.5,1.5,.5]);y=np.array([0,1,1,0]);threshold=.5
    probability,decision=source_decisions(logits,threshold,2.)
    np.testing.assert_array_equal(decision,[False,False,True,False])
    result=classification(y,logits,threshold)
    assert result['review_rate']==decision.mean() and result['accuracy']==(decision==y).mean()
    assert probability.shape==logits.shape and (probability>0).all() and (probability<1).all()
    with pytest.raises(ValueError):source_decisions(logits,threshold,0.)


def test_numerical_signature_verifier_detects_prediction_or_localization_change():
    values={'source':{'logit':.5,'native_samples':np.array([.2,.7],np.float32)}}
    assert verify_signatures(values,copy.deepcopy(values))['logit_max_absolute_error']==0
    changed=copy.deepcopy(values);changed['source']['native_samples'][0]=.4
    with pytest.raises(AssertionError):verify_signatures(values,changed)
    with pytest.raises(RuntimeError,match='identities'):verify_signatures(values,{})


def test_complete_efficiency_schedule_covers_every_predeclared_architecture(monkeypatch):
    import run_inference_efficiency as module
    monkeypatch.setattr(module,'read_json',lambda path:{'grid':[{'family':f'feature{i}'} for i in range(15)]})
    jobs=module.model_schedule()
    assert len(jobs)==97 and len({(j['family'],j['seed']) for j in jobs})==97
    assert sum(j['source_view']=='ensemble' for j in jobs)==3
    assert sum(j['kind']=='classical' for j in jobs)==21
    assert all(j['seed']==11 for j in jobs if j['kind']=='classical')
    assert sum(j['family'].endswith('_b20') for j in jobs)==18  #15 memory+3 matched supervised published cases.


def test_benchmark_verifier_requires_actual_full_pass_and_numerical_signature_coverage(tmp_path,monkeypatch):
    from elresearch import efficiency_evidence as module
    from elresearch import confirmation_registry as gate
    from elresearch.common import save_json,read_json
    monkeypatch.setattr(gate,'ROOT',tmp_path);monkeypatch.setattr(module,'ROOT',tmp_path)
    source=tmp_path/'source';source.mkdir();save_json(source/'run.json',{'fixture':True})
    study=tmp_path/'efficiency';study.mkdir();job_id='fixture_seed11';destination=study/'jobs'/job_id;destination.mkdir(parents=True)
    rows=source_image_rows(fixture_rows(tmp_path),['0','1','2'])
    settings={'study_id':'fixture','planned_jobs':1,'cohorts':{'fixture':rows},'batch_sizes':[1,2],
        'warmup_batches':1,'repetitions':2,'seed':1,'memory_sampling_interval':.01,
        'models':[{'kind':'dense','family':'fixture','seed':11,'source_view':'primary','run_folder':'source'}]}
    save_json(study/'protocol.json',settings)
    binding={'kind':'dense','family':'fixture','seed':11,'source_view':'primary','run':gate.bind_file(source/'run.json'),
        'source_protocol':gate.bind_file(study/'protocol.json')}
    save_json(study/'model_bindings.json',{'protocol_sha256':sha256(study/'protocol.json'),'jobs':[{'id':job_id,'binding':binding}]})
    monkeypatch.setattr(module,'bind_model',lambda *a:binding)
    class Dense(ImageScorer):
        def __call__(self,batch):return {**super().__call__(batch),'crack_probability':batch['images'][:,0].sigmoid()}
    signature=destination/'fixture.npz'
    measurements=measure_cohort(Dense(),rows,settings,0.,1.,'fixture',signature)
    result={'model_id':job_id,'study_id':'fixture','protocol_sha256':sha256(study/'protocol.json'),
        'model_bindings_sha256':sha256(study/'model_bindings.json'),'binding':binding,'prediction_equivalence_verified':True,
        'cohorts':{'fixture':measurements},'numerical_signature_files':{'fixture':gate.bind_file(signature)}}
    save_json(destination/'result.json',result);save_json(study/'status.json',{'state':'complete','completed_jobs':1})
    assert module.verify_completed_benchmark(study)['verified_jobs']==1
    result['cohorts']['fixture'][0]['repeated_uninstrumented_passes'][0]['images']=2
    save_json(destination/'result.json',result)
    with pytest.raises(RuntimeError,match='complete cohort'):module.verify_completed_benchmark(study)
