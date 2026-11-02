"""Concrete source-reviewed full-scope specification; no freeze or inference.

Explicitly retain all1,061original candidate records. Contrast edges below follow
the original questions/controls, rather than a target-result or source-rank filter.
All stored outcomes/failures remain required even for non-primary family pairs.
"""
import collections
import copy
import math
from pathlib import Path
import shutil

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch import confirmation_registry as gate
from elresearch.confirmation_workflow import specification_template,compile_jobs,validate_primary_comparisons,propose_fixed_examples,validate_fixed_examples
from elresearch.confirmation_population import checked_policy,selected_rows

BASE=ROOT/'artifacts/publication_20261017'
OUT=BASE/'final_source_review_v1/concrete_specification_v1'
CATALOG=BASE/'confirmation_preparation_v1/complete_source_candidate_catalog_20261027_v2.json'


def ref(path):
    return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha256(path)}


def main():
    if OUT.exists():raise FileExistsError('Use a registered new version instead of replacing concrete choices')
    assert not (BASE/'final_confirmation_v1/frozen_registry.json').exists()
    assert not (BASE/'final_confirmation_v1/observation_opened.json').exists()
    review_path=BASE/'final_source_review_v1/complete_source_review/review.json'
    review=read_json(review_path)
    assert review['new_source_evidence_review_complete'] and review['model_scope_decision']['retain_all1061_catalogued_records']
    catalog=read_json(CATALOG);candidates=catalog['candidates']
    audit=read_json(BASE/'final_source_review_v1/candidate_identity_audit_v2/results.json')
    models=[];by_family=collections.defaultdict(list);units=collections.defaultdict(dict)
    source_bindings={}
    candidate_by_family={}
    for c in candidates:
        row=next(r for r in audit['models'] if r['id']==c['id'])
        assert sha256(ROOT/row['binding']['path'])==row['binding']['sha256']
        m=read_json(ROOT/row['binding']['path']);models.append(m);by_family[m['family']].append(m)
        key=(c['source_dataset'],c['fold'],c['training_fraction'])
        semantic=(c['method'],c['inference_view'])
        if semantic in units[key]:assert units[key][semantic]==m['family'],(key,semantic)
        units[key][semantic]=m['family'];candidate_by_family[m['family']]=c
        source_bindings[m['id']]=row['binding']
    assert len(models)==1061 and len(by_family)==427
    edges={}
    def add(unit,baseline,experimental,purpose,baseline_view='primary',experimental_view='primary'):
        methods=units[unit]
        a=methods.get((baseline,baseline_view));b=methods.get((experimental,experimental_view))
        if a is None or b is None or a==b:return
        unordered=tuple(sorted((a,b)))
        if unordered in edges:
            if purpose not in edges[unordered]['purposes']:edges[unordered]['purposes'].append(purpose)
            return
        edges[unordered]={'first_family':a,'second_family':b,'purposes':[purpose],'unit':list(unit)}
    for unit,methods in units.items():
        for control in ['dinov3_frozen','dinov3_full','resnet18_full','resnet50_full','efficientnet_b0_full']:
            add(unit,control,'dinov3_lora','Original image-only LoRA versus frozen/full/CNN systems')
        for enc in ['resnet18','resnet50','efficientnet_b0']:
            for rate in ['1e-05','0_0003']:
                add(unit,enc+'_full_lr'+rate,'dinov3_lora','Post-hoc source image-only CNN optimization sensitivity, frozen prospectively for confirmation')
        for classifier in ['logistic','linear_svm','rbf_svm']:
            add(unit,'hog_stats_'+classifier,'dinov3_cls_'+classifier,'Same-classifier frozen DINO versus HOG/statistics feature control')
            add(unit,'hog_stats_'+classifier,'dinov3_lora','Original image-only LoRA versus classical control; different tuning budgets')
            add(unit,'dinov3_cls_'+classifier,'dinov3_lora','Original neural adaptation versus frozen DINO/classical classifier')
        for size in [224,336,448]:
            for pool in ['cls','patch_mean','concat']:
                method=f'dinov3_small_{size}_{pool}_logistic'
                add(unit,'resnet18_336_global_logistic',method,'Frozen representation complete-system comparison; pretraining/geometry confounded')
                if size!=336:add(unit,f'dinov3_small_336_{pool}_logistic',method,'Processing-scale control; no recovered physical-detail claim')
                if pool!='cls':add(unit,f'dinov3_small_{size}_cls_logistic',method,'CLS versus patch-mean/concatenation at the same processing size')
        for pool in ['cls','patch_mean','concat']:
            method=f'dinov3_base_336_{pool}_logistic'
            add(unit,'resnet18_336_global_logistic',method,'Frozen larger-encoder complete-system control')
            add(unit,f'dinov3_small_336_{pool}_logistic',method,'Model-scale control; pretraining and parameter count are not architecture-only causes')
        for classifier in ['linear_svm','rbf_svm']:
            add(unit,'resnet18_336_global_'+classifier,'dinov3_cls_'+classifier,'Same-classifier ImageNet ResNet versus DINO frozen-feature control')
        ref_method='ucf_reference_position' if ('ucf_reference_position','primary') in methods else 'dinov3_dense_reference'
        for control in ['deeplabv3plus_full','dinov3_dense_frozen','dinov3_dense_full',ref_method]:
            add(unit,control,'dinov3_dense_lora','Original dense adaptation and same-mask CNN/frozen/full/reference controls')
        add(unit,'dinov3_dense_frozen',ref_method,'Normal-reference versus frozen dense; matching-specific cause not isolated')
        for variant in ['dinov3_reference_no_position','dinov3_reference_mean_evidence','ucf_reference_no_position','ucf_reference_mean_evidence']:
            add(unit,variant,ref_method,'Predeclared geometry or top5%-versus-mean evidence control')
        for control in ['dinov3_dense_frozen','dinov3_dense_lora','dinov3_dense_full','deeplabv3plus_full']:
            add(unit,control,'ucf_reference_position','Original UCF-only reference future-transfer comparison')
        add(unit,'dinov3_dense_frozen','dinov3_patch_mlp','Local-patch objective/decoder control, same annotation identities when verified')
        add(unit,'dinov3_dense_lora','dinov3_patch_mlp','Adapted multidepth decoder versus local frozen-patch system')
        for control in ['dinov3_dense_frozen',ref_method]:
            add(unit,control,'dinov3_global_normal_distance','Fixed source-calibrated normal distance versus supervised dense evidence; cutoff conventions differ')
        add(unit,'dinov3_dense_lora','dinov3_postlora_no_reference','Fresh decoder after adapted-encoder freezing; extra optimization is part of the system')
        add(unit,'dinov3_postlora_no_reference','dinov3_postlora_reference','Matched fresh decoder reference versus no-reference')
        add(unit,'dinov3_dense_lora','dinov3_postlora_reference','Full post-adaptation system versus original parent, including adverse cells')
        for control in ['deeplabv3plus_full','dinov3_dense_frozen','dinov3_dense_lora','dinov3_dense_full',ref_method,
                        'dinov3_frozen_ucf20','dinov3_lora_ucf20','dinov3_reference_ucf20']:
            add(unit,'published_deeplab_fsl',control,'Complete supervised local published-architecture port versus declared dense comparator')
        for view in ['primary','ensemble']:
            add(unit,'published_deeplab_fsl','published_psmt_ssl','Semi-supervised local port versus full supervised architecture; unlabelled exposure retained',experimental_view=view)
            for control in ['dinov3_dense_frozen','dinov3_dense_lora',ref_method,'dinov3_frozen_ucf20','dinov3_lora_ucf20','dinov3_reference_ucf20']:
                add(unit,'published_psmt_ssl',control,'Dense comparator versus source-selected semi-supervised local port; primary and ensemble correlated',baseline_view=view)
        add(unit,'published_psmt_ssl','published_psmt_ssl','Correlated teacher-ensemble view versus released primary teacher1',experimental_view='ensemble')
        for view in ['primary','ensemble']:
            add(unit,'published_psmt_ssl','published_ssl_literal_xi1e6','Predeclared literal numerical recipe sensitivity versus selected port; not xi-only causal comparison',baseline_view=view,experimental_view=view)
        add(unit,'published_ssl_literal_xi1e6','published_ssl_literal_xi1e6','Correlated ensemble versus primary within literal numerical sensitivity',experimental_view='ensemble')
        for control in ['dinov3_frozen_ucf20','dinov3_reference_ucf20']:
            add(unit,control,'dinov3_lora_ucf20','All matched UCF20% DINO control systems, same86training+144validation masks')
        add(unit,'dinov3_frozen_ucf20','dinov3_reference_ucf20','Matched UCF20% frozen decoder versus source normal reference')
    # Every original family appears in at least one explicit control question;
    # no disconnected favorable/failed family can silently disappear.
    used={x[k] for x in edges.values() for k in ['first_family','second_family']}
    assert used==set(by_family), sorted(set(by_family)-used)
    declarations=[copy.deepcopy(c['declaration']) for c in candidates]
    jobs=compile_jobs(declarations,models)
    assert len(jobs)==1683
    contrasts=[];ledger=[]
    for edge in edges.values():
        a,b=edge['first_family'],edge['second_family'];am,bm=by_family[a],by_family[b]
        ca,cb=candidate_by_family[a],candidate_by_family[b]
        cohorts=set(ca['declaration']['cohorts'])&set(cb['declaration']['cohorts'])
        stratum=ca['declaration']['analysis_stratum']
        assert stratum==cb['declaration']['analysis_stratum']
        spatial=all(m['kind']=='dense' for m in am+bm)
        aa={m['seed']:m for m in am};bb={m['seed']:m for m in bm}
        annotation_match=True
        for seed in set(aa)|set(bb):
            x=aa.get(seed,am[0]);y=bb.get(seed,bm[0])
            annotation_match &= set(x['source_train_ids'])==set(y['source_train_ids']) and set(x['source_validation_ids'])==set(y['source_validation_ids'])
        for co in sorted(cohorts):
            endpoints=(['known_crack_recall','author_normal_FPR']+(['box_point_hit_rate'] if spatial else []) if co.startswith('pvelad') else
                       ['AUROC','average_precision','recall_at_source_threshold','review_rate_at_source_threshold','false_positive_rate_at_source_threshold']+
                       (['positive_image_Dice','positive_image_IoU'] if spatial else [])+
                       (['normal_structure_pixel_FPR'] if spatial and co=='scdd_publication_v1' else [])+
                       (['pooled_native_pixel_AP'] if spatial and stratum=='whole' else []))
            for endpoint in endpoints:
                row={'first_family':a,'second_family':b,'cohort':co,'endpoint':endpoint,
                     'analysis_stratum':stratum,'matched_annotation_identities':bool(annotation_match)}
                row['id']='contrast_'+digest(row)[:24]
                contrasts.append(row)
                ledger.append({'id':row['id'],'purposes':edge['purposes'],'unit':edge['unit'],
                               'difference_direction':'second_family minus first_family',
                               'same_annotation_identities':bool(annotation_match),
                               'same_annotation_ids_do_not_imply_equal_objectives_HPO_unlabelled_use_or_compute':True})
    validate_primary_comparisons(contrasts,models,jobs)
    spec=specification_template()
    spec.update({'source_review_completed':True,'models':declarations,'primary_comparisons':contrasts,
                 'primary_endpoints':sorted({c['endpoint'] for c in contrasts}),'inference_device':'mps',
                 'storage':{'minimum_free_GiB':16.,'retire_verified_native_pixel_cache':True}})
    spec['source_review']={'rationale':
        'All original fitting,69published-method,97formal efficiency and24ELPVouter gates are complete. The NEW completed source review retains every one of1061originally catalogued source-selected/development records in427recipe families, including all stronger CNN-rate sensitivities, literal numerical-port sensitivities, budgets and adverse seeds. This is complete-system evaluation, not a source-selected universal winner or a causal architecture/matching experiment. Original candidate cohort/held-stratum scope is preserved. Every declared model/cohort job and primary endpoint contrast must be reported; sensitivities and correlated views are not independent replications. Neither UCF nor SCDD is a wholly unobserved study-design domain. No target outcome enters these decisions.',
        'inputs':[str(p.relative_to(ROOT)) for p in [review_path,BASE/'final_source_review_v1/complete_source_review/registration.json',
                  BASE/'final_source_review_v1/complete_source_review/all_source_family_endpoints.json',
                  BASE/'final_source_review_v1/complete_source_review/all1061_source_candidate_endpoints.csv',
                  BASE/'final_source_review_v1/complete_source_review/all10_dense_cell_contrasts.json',
                  BASE/'final_source_review_v1/complete_source_review/all6_published_held_cell_contrasts.json',
                  BASE/'final_source_review_v1/candidate_identity_audit_v2/results.json',
                  BASE/'confirmation_preparation_v1/complete_source_candidate_catalog_20261027_v2_verification.json',
                  OUT/'contrast_rationale.json',OUT/'storage_and_example_policy_review.json']]}
    spec['limits'] += [
        'All paired intervals are conditional on these fitted systems and unadjusted for the full family of comparisons. They do not establish adjusted significance or training-population uncertainty.',
        'Image-only, dense-fusion, adapted CNN and full published-architecture ports are separate systems. Matched annotation identities do not match pretraining, loss, optimizer work or unlabelled exposure.',
        'CNN learning-rate and literal-VAT controls remain post-hoc/source numerical sensitivities. Their target contrasts are fixed before scoring; no target sensitivity selects a replacement model.',
        'Normal distance uses its frozen source normal-pixel cutoff, whereas posterior models use0.5; Dice differences do not isolate ranking quality or a matching-specific cause.',
        'Source threshold,temperature,normal memory,preprocessing,checkpoint and score direction remain fixed even for empty or below-chance outcomes.',
        'Source-held model jobs score the unchanged117pool but primary contrasts use only the matching acquisition stratum. Their pooled native AP remains a whole-pool descriptive point, never a stratified AP inference.',
        'ELPV1968review-label CV is a separate supplement. All656course-test inference remains forbidden.',
        'Formal cost evidence covers the97registered source model/seed cases, not new independent timings of every confirmation checkpoint or target image.',
        'Main article organization follows the original adaptation/label-efficiency/localization/source-transfer questions. All remaining frozen control contrasts and all models remain in supplements; no favorable target-driven model or endpoint selection.']
    population=checked_policy(ref(ROOT/spec['cohort_analysis_policy']))
    cohorts=[gate.cohort_binding(name) for name in sorted(gate.ALLOWED_COHORTS)]
    old_examples_path=BASE/'confirmation_preparation_v1/metadata_only_example_proposal.json'
    old=read_json(old_examples_path)
    examples={};changes=[];geometry=[]
    for co in cohorts:
        rows=[r for r in read_json(gate.check_file(co['manifest']))['records'] if r['partition']==co['partition']]
        primary,_=selected_rows(rows,co['name'],population)
        allowed={r['id'] for r in primary}
        previous=old['fixed_localization_examples'][co['name']]
        valid=(2<=len(previous)<=12 and len(set(previous))==len(previous) and set(previous)<=allowed)
        examples[co['name']]=previous if valid else propose_fixed_examples(primary)
        changes.append({'cohort':co['name'],'old_metadata_proposal_retained':valid,
                        'old_ids_outside_primary':sorted(set(previous)-allowed),
                        'reason':'Retain original metadata-only proposal if prospective primary-compatible; otherwise apply original deterministic class/channel/ID policy to primary metadata only. No pixels or outcomes.'})
        pixels=sum(math.prod(r['shape'][:2]) for r in rows)
        geometry.append({'cohort':co['name'],'scored_images':len(rows),'primary_images':len(primary),
                         'native_pixels_from_existing_manifest_shapes':pixels,
                         'would_use_per_pixel_AP_cache':not co['name'].startswith('pvelad'),
                         'uncompressed_bool_and_float32_cache_GiB_upper_bound':pixels*5/2**30 if not co['name'].startswith('pvelad') else 0.,
                         'PVEL_annotation_kind':'boxes/author-normal; never a full-cohort pixel AP cache' if co['name'].startswith('pvelad') else None})
    spec['fixed_localization_examples']=examples
    validate_fixed_examples(examples,models,jobs,cohorts,population)
    assert shutil.disk_usage(ROOT).free/2**30>spec['storage']['minimum_free_GiB']
    OUT.mkdir(parents=True)
    save_json(OUT/'contrast_rationale.json',ledger)
    save_json(OUT/'storage_and_example_policy_review.json',{
        'at':now(),'free_GiB_at_review':shutil.disk_usage(ROOT).free/2**30,'minimum_free_GiB_before_each_job':16.,
        'cohort_geometry':geometry,'old_fixed_example_proposal':ref(old_examples_path),'fixed_example_decisions':changes,
        'unchanged_evaluator_review':{
            'native_accumulator':ref(ROOT/'elresearch/confirmation_evaluate.py'),
            'streaming':ref(ROOT/'elresearch/streaming_inference.py'),
            'scorer':ref(ROOT/'elresearch/confirmation_predict.py'),
            'retirement_and_execution':ref(ROOT/'elresearch/confirmation_workflow.py'),
            'limits':['Scorer receives images/geometry only and uses frozen weights and source operating points.',
                      'Native pooled AP finish concatenates all SCDD/UCF known pixels in RAM, with sklearn scratch beyond the listed bool/float32 payload. This is bounded-cohort evaluation, not constant-memory exact AP.',
                      'Largest UCF native-cache bool/float32 payload is under0.35GiB; duplicate concatenation alone under0.70GiB, excluding AP sorting/workspace and other tensors.',
                      'PVEL12788695040native pixels never enter a cohort-wide AP cache. Each map is consumed to box/author-normal summaries; only fixed example maps persist.',
                      'Per-job derived pixel cache is retired only after saved prediction/result and every per-image cache identity/count/score check. Rawdata,bestweights,predictionarrays,nativecount summaries and fixed maps are retained.',
                      'Disk is checked before every job, not reserved. An actual resource failure counts as observed and permits exact frozen continuation only; no changed model/cohort/threshold/evaluator.']},
        'no_target_pixels_or_predictions_read':True,'final_registry_created':False})
    save_json(OUT/'specification.json',spec)
    save_json(OUT/'verification.json',{
        'at':now(),'builder':ref(Path(__file__)),'specification':ref(OUT/'specification.json'),
        'source_review':ref(review_path),'catalog':ref(CATALOG),'candidate_audit':ref(BASE/'final_source_review_v1/candidate_identity_audit_v2/results.json'),
        'models':len(models),'families':len(by_family),'jobs':len(jobs),'family_pair_questions':len(edges),
        'primary_contrast_endpoint_declarations':len(contrasts),
        'counts_by_cohort':dict(collections.Counter(c['cohort'] for c in contrasts)),
        'counts_by_endpoint':dict(collections.Counter(c['endpoint'] for c in contrasts)),
        'all427families_have_explicit_control_questions':used==set(by_family),
        'original_primary_validator_passed':True,'original_fixed_example_validator_passed':True,
        'annotation_identity_match_flags_computed_from_exact_source_bindings':True,
        'preserved_grouping_failure':ref(BASE/'final_source_review_v1/candidate_grouping_amendment_v2/audit_failure_v1.json'),
        'prepared_plan_created':False,'final_registry_created':False,'confirmation_outcomes_opened':False,
        'next':'Use unchanged run_confirmation_workflow.py prepare, review its actual complete bindings and coverage, then freeze the ONEcanonical registry; no default inference.'})
    print('CONCRETE SPECIFICATION',len(models),'models',len(jobs),'jobs',len(contrasts),'endpoint contrasts; no registry or scoring',flush=True)


if __name__=='__main__':main()
