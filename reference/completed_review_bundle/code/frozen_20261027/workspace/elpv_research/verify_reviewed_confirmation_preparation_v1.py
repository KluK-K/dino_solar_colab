"""Audit the actual prepared plan before its ONEcanonical freeze; no scoring."""
import collections
from pathlib import Path

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch import confirmation_registry as gate
from elresearch.confirmation_workflow import checked_preparation,validate_primary_comparisons,validate_fixed_examples
from elresearch.confirmation_population import checked_policy

BASE=ROOT/'artifacts/publication_20261017'
OUT=BASE/'final_source_review_v1/concrete_specification_v1'


def ref(path):
    return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha256(path)}


def main():
    destination=OUT/'prepared_plan_review.json'
    if destination.exists():raise FileExistsError('Preserve completed pre-freeze reviews')
    assert not (BASE/'final_confirmation_v1/frozen_registry.json').exists()
    assert not (BASE/'final_confirmation_v1/observation_opened.json').exists()
    plan=checked_preparation(OUT/'prepared_plan.json')
    spec=read_json(OUT/'specification.json')
    catalog=read_json(BASE/'confirmation_preparation_v1/complete_source_candidate_catalog_20261027_v2.json')
    audit=read_json(BASE/'final_source_review_v1/candidate_identity_audit_v2/results.json')
    audited={r['id']:r for r in audit['models']}
    models=plan['models'];jobs=plan['analysis_plan']['confirmation_jobs']
    assert {m['id'] for m in models}=={c['id'] for c in catalog['candidates']}==set(audited)
    assert spec['models']==[c['declaration'] for c in catalog['candidates']]
    for model in models:
        binding=audited[model['id']]['binding']
        assert sha256(ROOT/binding['path'])==binding['sha256']
        assert model==read_json(ROOT/binding['path']),model['id']
    assert len(models)==1061 and len({m['family'] for m in models})==427
    assert len({(j['model_id'],j['cohort']) for j in jobs})==len(jobs)==1683
    counts=collections.Counter(j['cohort'] for j in jobs)
    assert counts=={'scdd_publication_v1':1061,'ucf_reported_real_confirmation_v1':311,'pvelad_restricted_confirmation_v2':311}
    contrasts=plan['analysis_plan']['primary_comparisons']
    assert contrasts==spec['primary_comparisons'] and len(contrasts)==6212
    validate_primary_comparisons(contrasts,models,jobs)
    population=checked_policy(plan['analysis_plan']['cohort_analysis_policy_binding'])
    validate_fixed_examples(plan['analysis_plan']['fixed_localization_examples'],models,jobs,plan['cohorts'],population)
    for name,expected in [('cohort_analysis_policy_binding','281d82914d31e074c51133732fca4581f0f0f4df9ba2411b443011aed19e485f'),
                          ('secondary_native_failure_policy_binding','0729292cb410c941f0c4092625d8490a75bd6837280fd997ad18a77e07b9e350')]:
        assert plan['analysis_plan'][name]['sha256']==expected
        gate.check_file(plan['analysis_plan'][name])
    assert plan['analysis_plan']['storage_policy']=={'minimum_free_GiB':16.,'retire_verified_native_pixel_cache':True}
    assert plan['analysis_plan']['inference_device']=='mps'
    for co in plan['cohorts']:
        assert co==gate.cohort_binding(co['name'])
    prepared_gate_counts={key:read_json(gate.check_file(row['status']))['state'] for key,row in plan['development_evidence'].items()}
    assert set(prepared_gate_counts.values())=={'complete'}
    source_review_bindings=plan['analysis_plan']['source_review_inputs']
    for binding in source_review_bindings:gate.check_file(binding)
    raw_spec_source={key:spec[key] for key in ['source_review_completed','source_review','models','primary_comparisons','primary_endpoints','fixed_localization_examples','storage','limits']}
    proof={'at':now(),'code':ref(Path(__file__)),'prepared_plan':ref(OUT/'prepared_plan.json'),
           'preparation_id':plan['preparation_id'],'specification':ref(OUT/'specification.json'),
           'source_candidate_audit':ref(BASE/'final_source_review_v1/candidate_identity_audit_v2/results.json'),
           'full_scientific_specification_digest':digest(raw_spec_source),
           'models':len(models),'recipe_families':427,'unique_original_run_folders':len({m['run']['path'] for m in models}),
           'unique_original_checkpoint_byte_identities':len({m['checkpoint']['sha256'] for m in models}),
           'jobs':len(jobs),'counts_by_cohort':dict(counts),'all6212primary_endpoint_contrasts_verified':True,
           'all1061original_candidate_model_bindings_exact':True,'all_original_development_stages_complete':prepared_gate_counts,
           'both_original_population_and_secondary_failure_policies_exact':True,
           'source_review_inputs_checked':len(source_review_bindings),
           'original_fixed_example_proposal_retained_for_all3cohorts':all(r['old_metadata_proposal_retained'] for r in read_json(OUT/'storage_and_example_policy_review.json')['fixed_example_decisions']),
           'ready_for_single_canonical_freeze':True,'final_registry_created':False,'target_predictive_outcomes_opened':False,
           'limits':['No weight inference in this audit. Complete original97case efficiency and24outer-ELPV evidence validated by unchanged preparation workflow.',
                     'Ready for registry freeze is not scientific superiority, completed confirmation, a final paper or journal acceptance.',
                     '6212contrasts are all pre-score declared and unadjusted; no post-score favorable subset or adjusted-significance claim.']}
    save_json(destination,proof)
    print('PREPARED PLAN REVIEW PASSED',plan['preparation_id'],len(models),len(jobs),len(contrasts),flush=True)


if __name__=='__main__':main()
