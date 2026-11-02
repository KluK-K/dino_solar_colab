"""Synthetic-only orchestration fixtures; no real confirmation inputs or scores."""
import copy
import numpy as np
import pytest
from elresearch import confirmation_workflow as workflow
from elresearch import confirmation_registry as gate
from elresearch.common import digest,read_json,save_json,sha256
from elresearch.metrics import classification
from elresearch.publication_train import probability_metrics


def test_unreviewed_template_never_opens_confirmation_or_reads_cohort(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'ROOT',tmp_path);monkeypatch.setattr(workflow,'FINAL',tmp_path/'final')
    path=tmp_path/'selection.json';save_json(path,workflow.specification_template())
    monkeypatch.setattr(gate,'cohort_binding',lambda *a:pytest.fail('No cohort access for an unreviewed template'))
    with pytest.raises(RuntimeError,match='source-evidence review'):workflow.prepare(path,tmp_path/'plan.json')
    assert not (tmp_path/'final').exists() and not (tmp_path/'plan.json').exists()


def test_incomplete_source_studies_stop_preparation_before_cohort_access(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'ROOT',tmp_path);monkeypatch.setattr(workflow,'FINAL',tmp_path/'final')
    spec=workflow.specification_template();spec.update(source_review_completed=True,
        source_review={'rationale':'Synthetic source evidence, not real research','inputs':[]},
        primary_comparisons=['fixture'],primary_endpoints=['AUROC'],models=[{'kind':'image'}])
    path=tmp_path/'selection.json';save_json(path,spec)
    monkeypatch.setattr(workflow,'development_evidence',lambda:{})
    def incomplete(*a):raise RuntimeError('Development study is incomplete')
    monkeypatch.setattr(gate,'validate_development_evidence',incomplete)
    monkeypatch.setattr(gate,'cohort_binding',lambda *a:pytest.fail('No blind data before source completion'))
    with pytest.raises(RuntimeError,match='incomplete'):workflow.prepare(path,tmp_path/'plan.json')


def test_explicit_jobs_reject_duplicate_cohort_and_invalid_source_stratum():
    model={'id':'fixture_seed11'};declaration={'cohorts':['scdd_publication_v1'],'analysis_stratum':'ARTS','analysis_role':'Held-source frozen-weight fixture'}
    result=workflow.compile_jobs([declaration],[model]);assert result[0]['analysis_stratum']=='ARTS'
    with pytest.raises(ValueError,match='Repeated'):workflow.compile_jobs([declaration,declaration],[model,model])
    bad={**declaration,'cohorts':['pvelad_restricted_confirmation_v2']}
    with pytest.raises(ValueError,match='only to audited SCDD'):workflow.compile_jobs([bad],[model])
    with pytest.raises(PermissionError):workflow.compile_jobs([{**declaration,'cohorts':['unknown']}],[model])


@pytest.fixture
def scored(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    folder=tmp_path/'final';job={'model_id':'fixture_seed11','cohort':'fixture','analysis_stratum':'whole'}
    destination=folder/'results'/job['model_id']/job['cohort'];destination.mkdir(parents=True)
    rows=[{'id':'a','partition':'test','target':0},{'id':'b','partition':'test','target':1}]
    manifest=tmp_path/'cohort.json';save_json(manifest,{'records':rows})
    model={'id':job['model_id'],'source_threshold':.5,'source_temperature':2.}
    registry={'registry_id':'synthetic-only','models':[model],
              'cohorts':[{'name':'fixture','manifest':gate.bind_file(manifest),'partition':'test'}]}
    logits=np.array([-1.,.5]);y=np.array([0,1])
    np.savez_compressed(destination/'predictions.npz',ids=np.array(['a','b']),y=y,logits=logits)
    result={'registry_id':registry['registry_id'],'model_id':model['id'],'cohort':'fixture',
        'predictions_sha256':sha256(destination/'predictions.npz'),'metrics':classification(y,logits,.5),
        'source_frozen_threshold':.5,'source_frozen_temperature':2.,'threshold_fitted_on_target':False,
        'source_temperature_probability_metrics':probability_metrics(y,logits,2.),
        'localization':{'native_pixel_per_image':[{'id':'a','known_pixels':4,'crack_pixels':0},
                                                 {'id':'b','known_pixels':4,'crack_pixels':1}]}}
    save_json(destination/'result.json',result)
    cache=destination/'native_pixel_cache';cache.mkdir()
    for row in rows:
        truth=np.array([row['target'],0,0,0],bool);score=np.array([.8,.2,.1,.1],np.float32)
        np.savez_compressed(cache/(digest(row['id'])+'.npz'),truth=truth,score=score)
    fixed=destination/'fixed_native_examples';fixed.mkdir();(fixed/'keep.npz').write_bytes(b'figure fixture to retain')
    return folder,registry,job,destination,result


def test_output_audit_catches_changed_source_threshold_or_calibration(scored):
    folder,registry,job,destination,result=scored
    workflow.verify_prediction_result(folder,registry,job)
    result['source_frozen_temperature']=1.;save_json(destination/'result.json',result)
    with pytest.raises(RuntimeError,match='temperature'):workflow.verify_prediction_result(folder,registry,job)


def test_only_verified_owned_pixel_cache_is_retired_and_fixed_maps_remain(scored):
    folder,registry,job,destination,result=scored
    proof=workflow.retire_native_cache(folder,registry,job)
    assert proof['status']=='retired' and len(proof['files'])==2
    assert not list((destination/'native_pixel_cache').glob('*.npz'))
    assert (destination/'fixed_native_examples/keep.npz').read_bytes()==b'figure fixture to retain'
    assert workflow.retire_native_cache(folder,registry,job)==proof


def test_unknown_cache_file_causes_refusal_before_any_removal(scored):
    folder,registry,job,destination,result=scored
    np.savez_compressed(destination/'native_pixel_cache/not_owned.npz',truth=[0],score=[.1])
    with pytest.raises(RuntimeError,match='identities'):workflow.retire_native_cache(folder,registry,job)
    assert len(list((destination/'native_pixel_cache').glob('*.npz')))==3


def test_interrupted_cache_retirement_resumes_only_exact_owned_remaining_files(scored):
    folder,registry,job,destination,result=scored
    path=next((destination/'native_pixel_cache').glob('*.npz'));data=path.read_bytes()
    proof=workflow.retire_native_cache(folder,registry,job);path.write_bytes(data)
    proof['status']='retirement_prepared';save_json(destination/'native_cache_retirement.json',proof)
    resumed=workflow.retire_native_cache(folder,registry,job)
    assert resumed['status']=='retired' and not list((destination/'native_pixel_cache').glob('*.npz'))


def test_primary_comparisons_reject_invented_endpoints_or_supervision_matching():
    first={'id':'first_seed11','family':'first','seed':11,'kind':'dense','training_fraction':1.,
           'source_train_ids':['t0'],'source_validation_ids':['v0']}
    second={**first,'id':'second_seed11','family':'second'}
    jobs=[{'model_id':m['id'],'cohort':'scdd_publication_v1'} for m in [first,second]]
    declaration={'id':'fixed-source-mask-contrast','first_family':'first','second_family':'second',
        'cohort':'scdd_publication_v1','endpoint':'pooled_native_pixel_AP','analysis_stratum':'whole','matched_annotation_identities':True}
    workflow.validate_primary_comparisons([declaration],[first,second],jobs)
    with pytest.raises(ValueError,match='identities differ'):
        workflow.validate_primary_comparisons([declaration],[first,{**second,'source_train_ids':['different']}],jobs)
    pvel={**declaration,'cohort':'pvelad_restricted_confirmation_v2'}
    with pytest.raises(ValueError,match='annotation/task'):workflow.validate_primary_comparisons([pvel],[first,second],jobs)
    with pytest.raises(ValueError,match='concrete'):workflow.validate_primary_comparisons(['unresolved future model choice'],[first,second],jobs)
    held_jobs=[{**j,'analysis_stratum':'ARTS'} for j in jobs]
    with pytest.raises(ValueError,match='matching held-source'):workflow.validate_primary_comparisons([declaration],[first,second],held_jobs)
    with pytest.raises(ValueError,match='whole scored cohort'):
        workflow.validate_primary_comparisons([{**declaration,'analysis_stratum':'ARTS'}],[first,second],held_jobs)
    workflow.validate_primary_comparisons([{**declaration,'endpoint':'positive_image_Dice','analysis_stratum':'ARTS'}],[first,second],held_jobs)


def test_candidate_catalog_keeps_image_only_cnn_controls_out_of_dense_pipeline(tmp_path):
    from build_confirmation_catalog import candidate_kind
    assert candidate_kind({'cached_augmented_features':False,'model':{'encoder':'resnet50'}},tmp_path)=='image'
    assert candidate_kind({'model':{'architecture':'dinov3_multidepth'}},tmp_path)=='dense'
    (tmp_path/'model.joblib').write_bytes(b'fixture')
    assert candidate_kind({'specification':{'classifier':'rbf_svm'}},tmp_path)=='classical'


def test_fixed_native_gallery_is_metadata_only_and_required_before_map_cache_retirement(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    rows=[{'id':f'{source}-{target}-{i}','acquisition_source':source,'target':target,'partition':'test'}
        for source in ['ARTS','CFVS','CSIR'] for target in [0,1] for i in range(4)]
    proposal=workflow.propose_fixed_examples(rows)
    assert len(proposal)==6 and proposal==workflow.propose_fixed_examples(rows[::-1])
    path=tmp_path/'gallery_manifest.json';save_json(path,{'records':rows})
    models=[{'id':'fixture','kind':'dense'}];jobs=[{'model_id':'fixture','cohort':'scdd'}]
    cohorts=[{'name':'scdd','partition':'test','manifest':gate.bind_file(path)}]
    with pytest.raises(ValueError,match='Predeclare'):workflow.validate_fixed_examples({},models,jobs,cohorts)
    workflow.validate_fixed_examples({'scdd':proposal},models,jobs,cohorts)
    with pytest.raises(ValueError,match='both annotated'):workflow.validate_fixed_examples({'scdd':[r['id'] for r in rows if r['target']==1][:6]},models,jobs,cohorts)
    with pytest.raises(ValueError,match='Predeclare'):workflow.validate_fixed_examples({'scdd':proposal+['unknown']},models,jobs,cohorts)


def test_pvel_metadata_gallery_does_not_invent_physical_groups():
    rows=[{'id':f'pvel-{y}-{i}','target':y,'split_group':None} for y in [0,1] for i in range(7)]
    assert workflow.propose_fixed_examples(rows)==['pvel-0-0','pvel-0-3','pvel-0-6','pvel-1-0','pvel-1-3','pvel-1-6']
