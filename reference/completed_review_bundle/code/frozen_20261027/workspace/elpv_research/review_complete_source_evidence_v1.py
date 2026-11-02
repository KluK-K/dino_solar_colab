"""New post-completion source review; no target prediction/model execution.

Review source outcomes, strongest competing explanations and the original scope
before writing a concrete confirmation specification. Retain every catalogued
source-selected/development control; do not select a favorable winner or seed.
"""
import collections
import csv
import html
import json
from pathlib import Path
import statistics

from elresearch.common import ROOT, read_json, save_json, sha256, now
from elresearch.confirmation_workflow import development_evidence
from elresearch import confirmation_registry as gate

BASE = ROOT/'artifacts/publication_20261017'
OUT = BASE/'final_source_review_v1/complete_source_review'
CATALOG = BASE/'confirmation_preparation_v1/complete_source_candidate_catalog_20261027_v2.json'


def ref(path):
    return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha256(path)}


def checked(binding):
    path = ROOT/binding['path']
    assert sha256(path)==binding['sha256'], binding['path']
    return path


def source_points(candidate):
    m = candidate['source_validation_metrics']
    p = {'image_AUROC':m['auc'],'image_average_precision':m['average_precision'],
         'image_macro_F1':m['macro_f1'],'image_balanced_accuracy':m['balanced_accuracy'],
         'image_recall_at_source_threshold':m['recall'],'image_review_fraction_at_source_threshold':m['review_rate'],
         'image_FPR_at_source_threshold':m['fp']/(m['fp']+m['tn']) if m['fp']+m['tn'] else None}
    local = candidate['source_validation_localization']
    if local:
        p.update({'pooled_native_pixel_AP':local['pooled_pixel_average_precision'],
                  'positive_image_Dice':local['mean_positive_image_dice'],
                  'positive_image_IoU':local['mean_positive_image_iou'],
                  'negative_field_pixel_FPR':local['mean_negative_image_false_positive_fraction'],
                  'normal_structure_pixel_FPR':local['pooled_negative_structure_false_positive_fraction']})
    return p


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed reviews; register a new version for an amendment')
    assert not (BASE/'final_confirmation_v1/frozen_registry.json').exists()
    assert not (BASE/'final_confirmation_v1/observation_opened.json').exists()
    candidates = read_json(CATALOG)['candidates']
    audit_dir = BASE/'final_source_review_v1/candidate_identity_audit_v2'
    status = read_json(audit_dir/'status.json')
    assert status['state']=='complete' and status['checked']==1061 and status['families']==427
    audit = read_json(checked(status['result']))
    assert audit['all_original_binding_checks_passed'] and audit['all_declared_source_identity_D4_family_overlaps_zero']
    audited = {r['id']:r for r in audit['models']}
    assert set(audited)=={c['id'] for c in candidates}
    original_scope = read_json(BASE/'scope_coverage.json')
    scope_inputs = {Path(p) for row in original_scope['items'] for p in row.get('evidence',[])}
    # All evidence named by the old coverage ledger remains present; the ledger
    # is historical and its once-pending training claims are not current status.
    assert all((ROOT/p).exists() for p in scope_inputs)
    evidence = development_evidence()
    gate.validate_development_evidence(evidence)
    required_sources = [
        BASE/'scope_coverage.json',BASE/'pre_confirmation_transfer_scope_v1.json',CATALOG,
        BASE/'confirmation_preparation_v1/complete_source_candidate_catalog_20261027_v2_verification.json',
        audit_dir/'results.json',BASE/'source_development_identity_v1.json',
        BASE/'model_selection_domain_exposure_audit.json',BASE/'annotation_ontology_and_extent_audit.json',
        BASE/'statistics_plan_v2.json',BASE/'localization_and_inspection_plan_v2.json',
        BASE/'confirmation_preparation_v1/cohort_analysis_policy_v1.json',
        BASE/'confirmation_preparation_v1/secondary_native_failure_policy_v1.json',
        BASE/'classical_development_analysis_v1/results.json',BASE/'image_failure_analysis_v1/results.json',
        BASE/'completed_source_controls_v1/09ce4e229e6cf00e6dca/results.json',
        BASE/'ucf_reference_readout_v1/70faca6801b93cfd6a82/results.json',
        BASE/'native_annotation_support_v1/4d08723bf613017e29d2/results.json',
        BASE/'dense_development_uncertainty_v1/index.json',
        BASE/'complete_mechanism_summary_v1/d8e7a402f39cfd77a3f4/results.json',
        BASE/'published_complete_internal_readout_v1/4044360fa3f4fee05274/results.json',
        BASE/'published_complete_development_readout_v1/index.json',
        BASE/'published_source_calibration_readout_v1/863dbc9a403078c97329/results.json',
        BASE/'published_vat_sensitivity_readout_v1/d702716c8a3561f73b5c/results.json',
        BASE/'published_output_integrity_v1/index.json',
        BASE/'matched_UCF_source_readout_v1/e7742c7573c4cb2022a9/results.json',
        BASE/'matched_UCF_source_readout_v1/e7742c7573c4cb2022a9/numerical_review.json',
        BASE/'inference_efficiency_v1/verification.json',
        BASE/'complete_efficiency_readout_v1/4e1f4c8dcffe5ff1c86f/results.json',
        BASE/'complete_efficiency_readout_v1/4e1f4c8dcffe5ff1c86f/numerical_review.json',
        BASE/'elpv_review_supplement_v1/protocol.json',BASE/'elpv_review_supplement_v1/results.json',
        BASE/'complete_ELPV_supplement_readout_v1/268150aa6a1f25e7196e/results.json',
        BASE/'complete_ELPV_supplement_readout_v1/268150aa6a1f25e7196e/numerical_review.json',
        BASE/'method_provenance_v1/review.json',BASE/'component_method_provenance_v1/review.json',
        BASE/'source_access_and_reproduction_v1/review.json',
        BASE/'CFVS20_training_diagnostics_v3/a104b6f9eddcea8bc2d8/results.json',
        ROOT/'manuscript/manuscript_v04_inputs.json']
    dense_index = read_json(BASE/'dense_development_uncertainty_v1/index.json')
    assert len(dense_index['complete_cells'])==10
    dense_cells = []
    for row in dense_index['complete_cells']:
        path = checked(row); required_sources.append(path)
        data = read_json(path)
        dense_cells.append({key:data[key] for key in ['fold','budget','images','positive_images','provided_capture_families','all_model_seed_points','comparisons','limits']})
    assert sum(len(c['comparisons']) for c in dense_cells)==62
    assert sum(len(x['endpoints']) for c in dense_cells for x in c['comparisons'])==372
    published_index = read_json(BASE/'published_complete_development_readout_v1/index.json')
    assert len(published_index['complete_cells'])==6 and not published_index['pending_cells']
    published_cells = []
    for row in published_index['complete_cells']:
        path = checked(row); required_sources.append(path)
        review_path = path.parent/'numerical_review.json'; required_sources.append(review_path)
        data = read_json(path)
        published_cells.append({'source':ref(path),'numerical_review':read_json(review_path),
                                'all_system_endpoints':data['systems'],'all_comparisons':data['comparisons']})
    for stage in evidence.values():
        required_sources.extend([checked(stage['protocol']),checked(stage['status'])])
    for study in ['calibration_v1','classical_controls_v1','representation_scale_v1','dense_development_v1','mechanistic_controls_v1','published_methods_v1']:
        required_sources.append(BASE/study/'selected_hyperparameters.json')
    source_bindings = [ref(p) for p in sorted(set(required_sources))]
    stage_counts = {key:{'state':read_json(checked(value['status']))['state'],'expected_runs':value['expected_runs'],
                         'original_gate_validated':True} for key,value in evidence.items()}
    families = collections.defaultdict(list)
    point_rows = []
    for c in candidates:
        original_binding = read_json(checked(audited[c['id']]['binding']))
        assert original_binding['run']['sha256']==c['run_sha256']
        points = source_points(c)
        row = {'candidate_id':c['id'],'family':c['declaration']['family'],'source_dataset':c['source_dataset'],
               'fold':c['fold'],'budget':c['training_fraction'],'method':c['method'],
               'inference_view':c['inference_view'],'seed':c['seed'],
               'validation_images':c['source_validation_metrics']['n'],
               'training_images':len(original_binding['source_train_ids']),
               'validation_identity_set_sha256':audited[c['id']]['source_and_selection_identity_set_sha256'],
               'checkpoint_sha256':c['checkpoint_declared_sha256'],**points}
        # This digest includes training/HPO identities too; keep its meaning explicit.
        row['union_source_and_selection_identity_set_sha256'] = row.pop('validation_identity_set_sha256')
        point_rows.append(row)
        families[row['family']].append(row)
    summaries = []
    for family,rows in sorted(families.items()):
        metric_names = set().union(*(source_points(c).keys() for c in candidates if c['declaration']['family']==family))
        summaries.append({'family':family,'seeds':[r['seed'] for r in rows],
                          'source_dataset':rows[0]['source_dataset'],'fold':rows[0]['fold'],'budget':rows[0]['budget'],
                          'method':rows[0]['method'],'inference_view':rows[0]['inference_view'],
                          'metrics':{key:{'seed_values':[r.get(key) for r in rows],
                                          'mean':statistics.mean([r[key] for r in rows if r.get(key) is not None]) if any(r.get(key) is not None for r in rows) else None,
                                          'missing':sum(r.get(key) is None for r in rows)} for key in sorted(metric_names)}})
    decisions = [
        {'topic':'Image-only adaptation and CNN optimization','supported':'Source performance depends strongly on optimizer settings. ResNet50 sensitivity mean AUROC0.968762 exceeds image-only LoRA0.956; dense and image-only systems are separate.',
         'alternative_explanations':['Learning rate and checkpoint selection on reused validation','Different upstream pretraining/objectives and HPO budgets'],
         'decision':'Keep all six image families, all classical/probe representations and the completed CNN-rate controls visible. Rate sensitivities are post-hoc development controls; their future target contrasts are frozen before observation.'},
        {'topic':'Dense label efficiency and acquisition transfer','supported':'All ten dense cells and372 conditional endpoint summaries are complete. Image discrimination can improve while native AP or crack Dice worsens; full SCDD adapted-CNN Dice0.198 versus LoRA0.149 is retained.',
         'alternative_explanations':['Crack annotation extent/source/content','Different losses, local supervision, pretraining and optimization','Small reusable validation populations and conditional seed selection'],
         'decision':'Retain all registered5/10/20/100% internal budgets,20/100% held-acquisition models, three fitted seeds and all original source-only dense families. No target score-direction flip, threshold optimization or choice of favorable acquisition channel.'},
        {'topic':'Normal references, geometry, pooling and post-adaptation heads','supported':'Reference benefits are mixed across cells and seeds; extra head training can worsen full-budget Dice. Equal fitted parameter counts do not identify a matching-specific cause.',
         'alternative_explanations':['Extra decoder optimization and encoder freezing','Source normal-memory coverage and annotation/content differences'],
         'decision':'Retain reference/no-reference, position/no-position, top5%/mean, patch MLP, normal distance and fresh post-adaptation comparisons. Bind every seed-specific memory/cutoff/parent exactly; group by recipe rather than these fitted identities.'},
        {'topic':'Complete published architecture and semi-supervision','supported':'All69fits/69saved-output audits complete. Full architecture, literal VAT sensitivity and correlated teacher/ensemble views are retained, including empty masks and below-chance source-transfer scores.',
         'alternative_explanations':['Ported classes, preprocessing, batch size/backend, score convention and early stopping','Teacher1 versus correlated ensemble inference','Different original published task and supervision'],
         'decision':'Keep FSL, SSLprimary, SSLensemble, literal numerical sensitivity and matched UCF20% controls. Describe local ports and all numerical adaptations; do not rank mismatched published scores or infer failure of the authors original training.'},
        {'topic':'Matched UCF source comparison','supported':'LoRA Dice0.404889 versus full-FSL-port0.307231, favorable in all three fits; native AP and review advantages are not uniform. Missing normal-structure endpoints remain null.',
         'alternative_explanations':['Reused144 validation masks select checkpoints and calibration','86training+144validation are nested inside574UCF HPO exposure','Different model objectives/optimization/pretraining'],
         'decision':'Keep all four matched systems and seeds, plus full-source UCF controls as separate supervision regimes. Do not combine their validation means as independent replications.'},
        {'topic':'Formal speed and resources','supported':'All97isolated cases/388conditions/1164passes complete. Dense LoRA mean SCDD batch1 latency31.73ms/image versus adapted CNN27.08, with mixed paired directions; fewer trainable parameters do not imply faster inference.',
         'alternative_explanations':['Condition order, filesystem warmth, thermals, unrelated desktop activity','Different preprocessing geometry and implementation'],
         'decision':'Use original measured costs and all seed values. Keep RSS lifetime highwater, sampled MPS tensor and driver counters separate; no summed unified-memory pools or guaranteed-peak claim. Preserve pause-contaminated historical solver times separately.'},
        {'topic':'ELPV supplement','supported':'All24outer fits complete on1968 reused development images. Frozen DINOCLS RBF mean AUROC0.904918 versus HOG0.838261, all four fold directions positive; higher recall also raises review fraction.',
         'alternative_explanations':['Overlapping outer training folds','Representation dimensionality and inherited pretraining','Expert-review truth differs from crack presence/segmentation'],
         'decision':'Separate supplement only. No pooled uncalibrated OOF AUROC, independent-CNN or crack-ground-truth claim;656course-test inference remains forbidden.480inner fits follow successful grids, not480separately audited checkpoints.'},
        {'topic':'Confirmation populations and statistical interpretation','supported':'Integrity audit covered15963frames. Prospective PVEL primary exclusion34members gives12292images; score12326once and derive all declared views from those predictions.',
         'alternative_explanations':['Unverified physical modules/cropped overlaps/pretraining membership','Study-wide UCF/SCDD development exposure even for UCF-only fitted weights','Limited capture families and unadjusted multiple comparisons'],
         'decision':'Bind both original population and secondary-failure policies. Keep PVEL no-family-CI/box-truth limits; native pooled AP is a cohort point, not pixel-bootstrap inference. Keep all endpoints including nulls, failures, recall and review together.'},
        {'topic':'Evidence and publication claims','supported':'Source/access reviews and same-host implementation proofs exist; manuscript47 compiled. They do not reproduce independent predictive performance or prove novelty/acceptance.',
         'alternative_explanations':['Same-host/base-Python proof versus portable research reproduction','Prior DINOv3/normal-memory work and licensing differences'],
         'decision':'No first-DINOv3/EL/normal-memory novelty claim. Finish ONEregistry, every job/contrast/audit, native figures/pages and authorized-input reproduction package before submission readiness.'}]
    OUT.mkdir(parents=True)
    save_json(OUT/'registration.json',{'at':now(),'code':ref(Path(__file__)),'inputs':source_bindings,
                                      'scope':'New review after all original fit/cost/supplement gates; no registry or predictive target access.'})
    save_json(OUT/'all_source_family_endpoints.json',summaries)
    keys = list(dict.fromkeys(k for row in point_rows for k in row))
    with (OUT/'all1061_source_candidate_endpoints.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(point_rows)
    save_json(OUT/'all10_dense_cell_contrasts.json',dense_cells)
    save_json(OUT/'all6_published_held_cell_contrasts.json',published_cells)
    result = {'at':now(),'registration':ref(OUT/'registration.json'),'stage_gate_review':stage_counts,
              'historical_scope_requirements_retained':len(original_scope['items']),
              'source_catalog':ref(CATALOG),'candidate_identity_audit':ref(audit_dir/'results.json'),
              'original_candidate_count':1061,'recipe_family_count':427,
              'stochastic_or_subset_three_seed_families':317,'deterministic_full_budget_singletons':110,
              'source_family_endpoints':ref(OUT/'all_source_family_endpoints.json'),
              'all_source_candidate_endpoints':ref(OUT/'all1061_source_candidate_endpoints.csv'),
              'all10_dense_cells':ref(OUT/'all10_dense_cell_contrasts.json'),
              'all6_published_held_cells':ref(OUT/'all6_published_held_cell_contrasts.json'),
              'scientific_decisions':decisions,
              'model_scope_decision':{'retain_all1061_catalogued_records':True,'additional_unselected_HPO_trials':False,
                                     'rationale':'Preserve every completed original-scope source-selected/development model, budget, sensitivity and inference view. Avoid selecting a favorable winner from reused development results; main and sensitivity roles must be explicit in the concrete specification. This is a reviewed scope decision rather than an automatic promotion of the old unreviewed catalog.',
                                     'planned_jobs_if_original_candidate_cohorts_retained':1683,
                                     'cohort_counts':{'scdd_publication_v1':1061,'ucf_reported_real_confirmation_v1':311,'pvelad_restricted_confirmation_v2':311},
                                     'teacher_ensemble_views_are_correlated':True},
              'new_source_evidence_review_complete':True,'concrete_final_specification_complete':False,
              'still_required_before_freeze':['Concrete family-pair/cohort/stratum/endpoint declarations with source-annotation matching audited',
                                              'Fixed metadata-only examples compatible with prospective exclusions',
                                              'Storage/memory operational review of the unchanged streaming evaluator',
                                              'Prepare and check one actual complete plan, then freeze ONEcanonical registry'],
              'inference_or_fit_performed':False,'confirmation_outcomes_opened':False,'final_registry_created':False,
              'submission_ready':False}
    save_json(OUT/'review.json',result)
    paragraphs=''.join('<section><h2>'+html.escape(d['topic'])+'</h2><p>'+html.escape(d['supported'])+'</p><p><b>Competing explanations:</b> '+html.escape('; '.join(d['alternative_explanations']))+'</p><p><b>Decision:</b> '+html.escape(d['decision'])+'</p></section>' for d in decisions)
    (OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Completed source evidence review</title><style>body{max-width:1000px;margin:40px auto;font:17px/1.6 system-ui;color:#183b48}section{border-top:1px solid #ccd;padding:12px 0}a{color:#17617a}</style><h1>Completed source evidence review</h1><p>All original gates complete;1,061 source candidate records in427 recipe families. Existing source outcomes are development evidence. No final registry or target model outcomes opened. Concrete contrast/example/storage specification remains pending.</p>'+paragraphs+'<p><a href="review.json">Review and exact inputs</a> · <a href="all1061_source_candidate_endpoints.csv">Every source candidate endpoint</a> · <a href="all_source_family_endpoints.json">Every family/seed</a></p></html>',encoding='utf-8')
    print('NEWsource review complete;1061candidate records/427families,10dense+6published-held cells. Concrete final specification still required.',flush=True)


if __name__=='__main__':
    main()
