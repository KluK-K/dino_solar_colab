"""Synthetic frozen-policy fixtures; no real confirmation images or outputs."""
import copy
import numpy as np
import pytest

from elresearch.common import save_json,sha256
from elresearch import confirmation_registry as gate
from elresearch.secondary_failure_analysis import checked_failure_policy,native_failure_summary


def policy_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    helper=tmp_path/'analyze_dense_source_localization.py';helper.write_text('# synthetic evidence fixture only\n')
    plan=tmp_path/'source_plan.json';save_json(plan,{'analysis_code_sha256':sha256(helper),
        'positive_area_bins':[0,.005,.02,1],'pixel_cutoff':.5,'source_bindings':[{} for _ in range(12)]})
    results=tmp_path/'source_results.json';save_json(results,{'status':'complete','plan_sha256':sha256(plan),
        'seed_subgroup_records':60,'families':[{'fits':[{'seed':seed} for seed in [11,22,33]]} for _ in range(4)]})
    bind=lambda p:{'path':p.name,'sha256':sha256(p)}
    policy={'version':1,'before_confirmation_model_inference':True,
        'positive_area_bin_edges':[0,.005,.02,1],'pixel_score_threshold':.5,
        'allowed_native_cohorts':['scdd_publication_v1','ucf_reported_real_confirmation_v1'],
        'evidence':{'source_plan':bind(plan),'source_results':bind(results),'helper':bind(helper)}}
    path=tmp_path/'policy.json';save_json(path,policy)
    return path,policy,helper


def test_secondary_failure_policy_locks_source_bins_and_bound_evidence(tmp_path,monkeypatch):
    path,policy,helper=policy_fixture(tmp_path,monkeypatch)
    assert checked_failure_policy({'path':path.name,'sha256':sha256(path)})==policy
    changed=copy.deepcopy(policy);changed['positive_area_bin_edges']=[0,.01,.02,1];save_json(path,changed)
    with pytest.raises(ValueError,match='source-fixed'):checked_failure_policy({'path':path.name,'sha256':sha256(path)})
    save_json(path,policy);helper.write_text('# changed fixture\n')
    with pytest.raises(RuntimeError,match='Frozen file changed'):checked_failure_policy({'path':path.name,'sha256':sha256(path)})


def record():
    rows=[]
    for name,cracks,dice in [('p',2,0.),('n',0,1.),('outside',80,1.)]:
        rows.append({'id':name,'known_pixels':1000,'crack_pixels':cracks,'dice':dice,
            'annotated_crack_fraction':cracks/1000,'pixel_average_precision':.1 if cracks else None,
            'negative_image_false_positive_fraction':.01 if not cracks else None})
    return {'ids':np.array(['p','n']),'y':np.array([1,0]),'logits':np.array([.3,.7]),
        'source_threshold':.3,'analysis_stratum':'ARTS','model':{'seed':11,'architecture':'dinov3_multidepth'},
        'localization':{'pixel_threshold':.5,'native_pixel_per_image':rows}}


def test_secondary_native_summary_respects_selected_stratum_and_keeps_empty_groups_null():
    policy={'allowed_native_cohorts':['scdd_publication_v1','ucf_reported_real_confirmation_v1'],
        'positive_area_bin_edges':[0,.005,.02,1],'pixel_score_threshold':.5}
    output=native_failure_summary(record(),'scdd_publication_v1',policy)
    assert output['analyzed_images']==2
    low,medium,large,whole,negative=output['subgroups']
    assert whole['images']==1 and whole['mean_positive_image_Dice']==0
    assert low['source_image_recall']==0 and low['zero_overlap_fraction']==1
    assert not output['image_detected_with_zero_overlap_ids']
    assert medium['images']==0 and medium['mean_positive_image_Dice'] is None
    assert large['images']==0 and large['zero_overlap_fraction'] is None
    assert output['confidence_interval'] is None
    assert native_failure_summary(record(),'pvelad_restricted_confirmation_v2',policy) is None


def test_distance_cutoff_semantics_and_original_pixel_operating_point_are_preserved():
    policy={'allowed_native_cohorts':['scdd_publication_v1','ucf_reported_real_confirmation_v1'],
        'positive_area_bin_edges':[0,.005,.02,1],'pixel_score_threshold':.5}
    value=record();value['model']['architecture']='dinov3_normal_distance'
    assert 'noncrack-quantile' in native_failure_summary(value,'scdd_publication_v1',policy)['pixel_response_semantics']
    value['localization']['pixel_threshold']=.3
    with pytest.raises(ValueError,match='operating point'):native_failure_summary(value,'scdd_publication_v1',policy)


def test_frozen_saved_output_analysis_includes_secondary_counts_without_new_inference(tmp_path,monkeypatch):
    from elresearch import confirmation_workflow as workflow
    from elresearch.confirmation_analysis import analyze
    from elresearch.metrics import classification
    from elresearch.publication_train import probability_metrics
    policy_path,policy,_=policy_fixture(tmp_path,monkeypatch)
    cohort='scdd_publication_v1';ids=np.array(['p','n','p2','n2']);y=np.array([1,0,1,0]);logits=np.array([.4,-.2,.5,-.1])
    rows=[{'id':str(i),'partition':'test','target':int(t),'split_group':g,'acquisition_source':'ARTS'}
        for i,t,g in zip(ids,y,['fixtureA','fixtureA','fixtureB','fixtureB'])]
    manifest=tmp_path/'manifest.json';save_json(manifest,{'records':rows})
    statistics=tmp_path/'statistics.json';save_json(statistics,{'repetitions':50,'seed':20261017})
    models=[{'id':name+'_seed11','family':name,'seed':11,'training_fraction':1.,'kind':'dense',
        'architecture':'dinov3_multidepth','source_threshold':.3,'source_temperature':1.} for name in ['first','second']]
    jobs=[{'model_id':m['id'],'cohort':cohort,'analysis_stratum':'whole','analysis_role':'synthetic fixture'} for m in models]
    comparison={'id':'fixture-auroc','first_family':'first','second_family':'second','cohort':cohort,
        'analysis_stratum':'whole','matched_annotation_identities':True,'endpoint':'AUROC'}
    registry={'models':models,'cohorts':[{'name':cohort,'partition':'test','manifest':gate.bind_file(manifest)}],
        'evaluator_files':[],'analysis_plan':{'confirmation_jobs':jobs,'primary_comparisons':[comparison],
            'statistics_plan_binding':gate.bind_file(statistics),'secondary_native_failure_policy_binding':gate.bind_file(policy_path)}}
    registry['registry_id']=gate.registry_digest(registry);folder=tmp_path/'final';folder.mkdir();save_json(folder/'frozen_registry.json',registry)
    native=[]
    for i,t in zip(ids,y):
        cracks=2 if i=='p' else 10 if i=='p2' else 0
        native.append({'id':str(i),'crack_positive':bool(t),'known_pixels':1000,'crack_pixels':cracks,
            'annotated_crack_fraction':cracks/1000,'dice':0. if i=='p' else .5 if i=='p2' else 1.,
            'iou':0. if i=='p' else .333 if i=='p2' else 1.,'pixel_average_precision':.1 if t else None,
            'negative_image_false_positive_fraction':0. if not t else None})
    audits=[]
    for model,job in zip(models,jobs):
        dest=folder/'results'/model['id']/cohort;dest.mkdir(parents=True)
        np.savez_compressed(dest/'predictions.npz',ids=ids,y=y,logits=logits)
        value={'registry_id':registry['registry_id'],'model_id':model['id'],'cohort':cohort,
            'predictions_sha256':sha256(dest/'predictions.npz'),'metrics':classification(y,logits,.3),
            'source_frozen_threshold':.3,'source_frozen_temperature':1.,'threshold_fitted_on_target':False,
            'source_temperature_probability_metrics':probability_metrics(y,logits,1.),
            'development_exposure':{},'task_scope':'synthetic fixture only',
            'localization':{'pixel_threshold':.5,'native_pixel_per_image':native,'pooled_native_pixel_AP':.5}}
        save_json(dest/'result.json',value);audits.append({**job,'result':gate.bind_file(dest/'result.json')})
    save_json(folder/'execution_audit.json',{'registry_id':registry['registry_id'],'verified_jobs':2,'jobs':audits})
    monkeypatch.setattr(workflow,'evaluate',lambda *args:pytest.fail('Saved-output analysis cannot initiate inference'))
    result=analyze(folder)
    assert len(result['primary_comparisons'])==1 and result['primary_comparisons'][0]['estimate']['mean_pair_difference']==0
    for job in result['jobs']:
        diagnostic=job['secondary_native_failure_analysis'];positive=next(r for r in diagnostic['subgroups'] if r['subgroup']=='all_positives')
        assert positive['images']==2 and positive['image_detected_with_zero_overlap_images']==1
        assert diagnostic['image_detected_with_zero_overlap_ids']==['p'] and diagnostic['confidence_interval'] is None
    assert analyze(folder)==result
