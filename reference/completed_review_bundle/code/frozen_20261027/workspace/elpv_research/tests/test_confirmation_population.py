"""Primary near-copy quarantine uses metadata, not the favorable model outcome."""
import copy
import numpy as np
import pytest

from elresearch import confirmation_registry as gate
from elresearch.confirmation_population import checked_policy, selected_rows
from elresearch.confirmation_analysis import load_record, restricted_summary, endpoint_value
from elresearch.common import save_json, read_json, sha256
from elresearch.metrics import classification
from elresearch.publication_train import probability_metrics


@pytest.fixture
def population(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    name='pvelad_restricted_confirmation_v2'; ids=[f'pvelad/{i}' for i in range(6)];y=[0,0,1,1,1,1]
    rows=[{'id':i,'target':label,'partition':'external','split_group':None,'target_line_crack':label}
          for i,label in zip(ids,y)]
    manifest=tmp_path/'cohort.json';save_json(manifest,{'records':rows})
    folder=tmp_path/'artifacts/publication_20261017/whole_frame_similarity_audit_v1';folder.mkdir(parents=True)
    strong=[{'first_id':ids[4],'second_id':ids[5]}];broad=strong+[{'first_id':ids[1],'second_id':ids[3]}]
    screen=folder/'result.json';save_json(screen,{'search_complete':True,'confirmation_model_inference':False,
        'strong_similarity_pairs':strong,'screened_pairs':broad})
    review=folder/'origin_review.json';save_json(review,{'model_scores_used':False,'pvel_decisions':strong})
    def subset(excluded):
        keep=[r for r in rows if r['id'] not in excluded]
        return {'excluded_ids':excluded,'images':len(keep),'positive':sum(r['target'] for r in keep),'negative':sum(1-r['target'] for r in keep)}
    policy={'version':1,'before_confirmation_model_inference':True,'evidence':[gate.bind_file(screen),gate.bind_file(review)],
        'cohorts':{name:{'manifest':gate.bind_file(manifest),'subsets':{'primary':subset(ids[4:]),'full_v2_pool':subset([]),
            'broader_similarity_screen':subset([ids[1],ids[3],ids[4],ids[5]])}}}}
    path=tmp_path/'population.json';save_json(path,policy)
    return tmp_path,name,rows,policy,path,manifest


def test_frozen_population_counts_and_exclusions_match_the_whole_data_audit(population):
    root,name,rows,policy,path,manifest=population
    assert checked_policy(gate.bind_file(path))==policy
    assert len(selected_rows(rows,name,policy)[0])==4
    assert len(selected_rows(rows,name,policy,'full_v2_pool')[0])==6
    assert len(selected_rows(rows,name,policy,'broader_similarity_screen')[0])==2
    changed=copy.deepcopy(policy);changed['cohorts'][name]['subsets']['primary']['excluded_ids'].append(rows[2]['id'])
    save_json(path,changed)
    with pytest.raises(ValueError,match='outcome-selected'):checked_policy(gate.bind_file(path))


def test_original_pool_and_source_data_do_not_disappear_when_primary_is_filtered(population):
    root,name,rows,policy,path,manifest=population
    job={'model_id':'fixture_seed11','cohort':name,'analysis_stratum':'whole'}
    model={'id':job['model_id'],'family':'fixture','seed':11,'kind':'dense','training_fraction':1.,
        'source_threshold':0.,'source_temperature':1.}
    registry={'registry_id':'synthetic fixture, never real confirmation','models':[model],
        'cohorts':[{'name':name,'manifest':gate.bind_file(manifest),'partition':'external'}],
        'analysis_plan':{'cohort_analysis_policy_binding':gate.bind_file(path)}}
    destination=root/'results'/model['id']/name;destination.mkdir(parents=True)
    y=np.array([r['target'] for r in rows]);logits=np.array([-2.,-1.,1.,2.,-3.,4.])
    np.savez_compressed(destination/'predictions.npz',ids=np.array([r['id'] for r in rows]),y=y,logits=logits)
    result={'registry_id':registry['registry_id'],'model_id':model['id'],'cohort':name,
        'predictions_sha256':sha256(destination/'predictions.npz'),'source_frozen_threshold':0.,
        'source_frozen_temperature':1.,'metrics':classification(y,logits,0.),'threshold_fitted_on_target':False,
        'source_temperature_probability_metrics':probability_metrics(y,logits),
        'localization':{'pooled_native_pixel_AP':.7,'box_per_image':[{'id':rows[i]['id'],'maximum_point_in_target_box':i!=4} for i in range(2,6)]}}
    save_json(destination/'result.json',result)
    primary=load_record(root,registry,job);full=load_record(root,registry,job,subset='full_v2_pool')
    assert len(primary['ids'])==4 and primary['underlying_scored_images']==6
    assert restricted_summary(primary)['known_crack_images']==2 and restricted_summary(primary)['known_crack_recall']==1.
    assert restricted_summary(full)['known_crack_images']==4 and restricted_summary(full)['known_crack_recall']==.75
    assert endpoint_value(primary,'box_point_hit_rate')==1. and endpoint_value(full,'box_point_hit_rate')==.75
    with pytest.raises(ValueError,match='whole-cohort'):endpoint_value(primary,'pooled_native_pixel_AP')
    np.testing.assert_array_equal(np.load(destination/'predictions.npz')['logits'],logits)
    assert read_json(destination/'result.json')==result


def test_final_analysis_reports_primary_and_both_sensitivities_without_extra_inference(population,monkeypatch):
    from elresearch.confirmation_analysis import analyze
    from elresearch import confirmation_workflow as workflow
    root,name,rows,policy,path,manifest=population
    monkeypatch.setattr(workflow,'evaluate',lambda *a:pytest.fail('Population analysis never starts inference'))
    models=[{'id':family+'_seed11','family':family,'seed':11,'kind':'dense','training_fraction':1.,
        'source_threshold':0.,'source_temperature':1.} for family in ['first','second']]
    jobs=[{'model_id':m['id'],'cohort':name,'analysis_stratum':'whole','analysis_role':'Synthetic population fixture'} for m in models]
    statistics=root/'statistics.json';save_json(statistics,{'repetitions':50,'seed':20261017})
    registry={'models':models,'cohorts':[{'name':name,'manifest':gate.bind_file(manifest),'partition':'external'}],
        'evaluator_files':[],'analysis_plan':{'cohort_analysis_policy_binding':gate.bind_file(path),
            'statistics_plan_binding':gate.bind_file(statistics),'confirmation_jobs':jobs,
            'primary_comparisons':[{'id':'fixed_recall','first_family':'first','second_family':'second',
                'cohort':name,'endpoint':'known_crack_recall','analysis_stratum':'whole','matched_annotation_identities':True}]}}
    registry['registry_id']=gate.registry_digest(registry);save_json(root/'frozen_registry.json',registry)
    y=np.array([r['target'] for r in rows]);logits=np.array([-2.,-1.,1.,2.,-3.,4.]);audited=[]
    for m,job in zip(models,jobs):
        destination=root/'results'/m['id']/name;destination.mkdir(parents=True)
        np.savez_compressed(destination/'predictions.npz',ids=np.array([r['id'] for r in rows]),y=y,logits=logits)
        result={'registry_id':registry['registry_id'],'model_id':m['id'],'cohort':name,
            'predictions_sha256':sha256(destination/'predictions.npz'),'source_frozen_threshold':0.,
            'source_frozen_temperature':1.,'metrics':classification(y,logits,0.),'threshold_fitted_on_target':False,
            'source_temperature_probability_metrics':probability_metrics(y,logits),'task_scope':'Synthetic restricted fixture',
            'development_exposure':{},'localization':None}
        save_json(destination/'result.json',result);audited.append({**job,'result':gate.bind_file(destination/'result.json')})
    save_json(root/'execution_audit.json',{'registry_id':registry['registry_id'],'verified_jobs':2,'jobs':audited})
    proof=analyze(root)
    for row in proof['jobs']:
        assert row['metrics']['n']==4 and row['primary_restricted_endpoints']['known_crack_images']==2
        assert row['primary_restricted_endpoints']['known_crack_recall']==1.
        assert row['population_sensitivities']['full_v2_pool']['metrics']['n']==6
        assert row['population_sensitivities']['broader_similarity_screen']['metrics']['n']==2
        assert row['line_crack_only_sensitivity']['metrics']['n']==4
    contrast=proof['primary_comparisons'][0]['estimate']
    assert contrast['mean_pair_difference']==0. and contrast['acquisition_family_percentile_ci95'] is None
    assert analyze(root)==proof
