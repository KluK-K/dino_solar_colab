"""Review every complete source fit/view, preserving threshold/empty-mask context."""
import csv
import html
import statistics

from elresearch.common import ROOT, now, read_json, save_json, sha256, digest

BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'published_complete_internal_readout_v1'


def main():
    if not read_json(BASE / 'inference_efficiency_v1/status.json')['state'].startswith('queued'):
        raise PermissionError('Source reporting must not compete with formal efficiency')
    latest = read_json(SOURCE / 'latest.json')
    result_path = ROOT / latest['path']
    assert sha256(result_path) == latest['sha256']
    result = read_json(result_path)
    plan = read_json(ROOT / result['plan']['path'])
    source_bindings = {x['path']: x['sha256'] for x in plan['input_bindings']}
    audit_path = next(ROOT / x['path'] for x in plan['input_bindings'] if x['path'].endswith('completed_output_integrity_index_snapshot.json'))
    audited = {x['run']: x for x in read_json(audit_path)['completed_published_fits_verified']}
    inputs = [latest, {'path': str((ROOT / result['plan']['path']).relative_to(ROOT)), 'sha256': result['plan']['sha256']}]
    selections = [BASE / 'published_methods_v1/selected_hyperparameters.json', BASE / 'dense_development_v1/selected_hyperparameters.json']
    for path in selections:
        inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    selected = {x['name']: x for x in read_json(selections[0])['selected']}
    identity = {'primary_result_sha256': sha256(result_path), 'builder_sha256': sha256(__file__),
                'selection_bindings': inputs[2:]}
    directory = SOURCE / 'reporting' / digest(identity)[:20]
    if (directory / 'report_inputs.json').exists():
        previous = read_json(directory / 'report_inputs.json')
        for value in previous['files']:
            assert sha256(ROOT / value['path']) == value['sha256']
        print(str((directory / 'report.html').relative_to(ROOT)))
        return
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in result['all_seed_views']:
        path = ROOT / value['run']
        assert sha256(path) == source_bindings[value['run']]
        run = read_json(path)
        if value['system'].startswith('published_'):
            receipt = audited[value['run']]
            assert sha256(ROOT / receipt['verification']) == receipt['sha256']
            original = read_json(ROOT / receipt['verification'])
            binding = next(x for x in original['identity']['source_files'] if x['path'] == value['run'])
            assert binding['sha256'] == sha256(path)
            specification = selected[run['model']['name']]
            for key in ['semi', 'learning_rate', 'vat_xi', 'negative_threshold']:
                if key in specification:
                    assert run['model'][key] == specification[key]
            evaluated = run['inference_views'][value['view']]
        else:
            evaluated = run
        native = evaluated['validation_localization']['per_image']
        positives = [x for x in native if x['crack_positive']]
        negatives = [x for x in native if not x['crack_positive']]
        empty = sum(x['predicted_crack_pixels'] == 0 for x in positives)
        zero_overlap = sum(x['predicted_crack_pixels'] - x['false_positive_pixels'] == 0 for x in positives)
        structural = sum(x.get('negative_image_normal_structure_pixels') or 0 for x in negatives)
        structural_fp = sum(x.get('negative_image_normal_structure_false_positive_pixels') or 0 for x in negatives)
        rows.append({'budget': value['budget'], 'system': value['system'], 'seed': value['seed'],
                     'positive_images': len(positives), 'zero_prediction_positive_images': empty,
                     'zero_overlap_positive_images': zero_overlap,
                     'zero_prediction_positive_fraction': empty/len(positives),
                     'zero_overlap_positive_fraction': zero_overlap/len(positives),
                     'predicted_crack_pixels_on_positives': sum(x['predicted_crack_pixels'] for x in positives),
                     'predicted_crack_pixels_on_negatives': sum(x['predicted_crack_pixels'] for x in negatives),
                     'eligible_negative_structure_pixels': structural,
                     'false_positive_negative_structure_pixels': structural_fp,
                     'normal_structure_FPR': structural_fp/structural if structural else None})
    diagnostic = {'at': now(), 'identity': identity, 'input_bindings': inputs,
                  'analysis_timing': 'Retrospective whole-cohort interpretation of the observed zero structural-FPR cells; every54 view/seed record and both budgets retained without subgroup/example selection.',
                  'rows': rows, 'unique_source_validation_images': 120,
                  'same_54_positives_reused_in_every_fit': True,
                  'native_cutoff_unchanged': .5, 'new_primary_endpoints_or_significance_tests': False,
                  'scope': 'Secondary original integer counts distinguish empty predictions from zero overlap. A zero normal-structure pixel FPR cannot establish good crack recovery or low image-level review workload.'}
    save_json(directory / 'native_failure_context.json', diagnostic)
    with (directory / 'all54_native_failure_records.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    sections = []
    for cell in result['complete_internal_cells']:
        table = []
        for system, values in cell['systems'].items():
            matching = [x for x in rows if x['budget'] == cell['budget'] and x['system'] == system]
            points = [values[x]['mean'] for x in ['AUROC', 'pooled_native_pixel_AP', 'positive_image_Dice',
                                                 'normal_structure_pixel_FPR', 'recall_at_source_threshold', 'review_rate_at_source_threshold']]
            cells = ['not estimable' if x is None else f'{x:.6f}' for x in points]
            empty = [str(x['zero_prediction_positive_images']) for x in matching]
            overlap = [str(x['zero_overlap_positive_images']) for x in matching]
            table.append('<tr><th>'+html.escape(system)+'</th>'+''.join('<td>'+x+'</td>' for x in cells)+
                         '<td>'+','.join(empty)+'</td><td>'+','.join(overlap)+'</td></tr>')
        sections.append('<h2>Training-mask fraction '+str(cell['budget'])+'</h2><div class="scroll"><table><tr><th>System/view</th><th>AUROC</th><th>Native AP</th><th>Positive Dice</th><th>Structure FPR ratio</th><th>Source recall</th><th>Review fraction</th><th>Empty positive counts s11,22,33</th><th>Zero-overlap counts s11,22,33</th></tr>'+''.join(table)+'</table></div>')
    report = '''<!doctype html><meta charset="utf-8"><title>Complete internal source comparison: endpoint and failure context</title>
<style>body{font:16px/1.55 system-ui;max-width:1600px;margin:32px auto;padding:20px;color:#183342}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:9px;border-bottom:1px solid #ccd7df;text-align:left;white-space:nowrap}.note{padding:16px;background:#edf4f6}</style>
<h1>Complete internal source comparisons: all models, all three seeds and both declared SSL views</h1>
<p class="note">Same120 reused validation images,54 crack positives and74 provided capture families. These are retrospective source-validation comparisons, without independent confirmation or confidence intervals. All table values are three-fit means; sample SD and every fitted point remain in the primary result/CSV. The two budgets reuse the same cohort and the teacher/ensemble share one fit.</p>
<p>At20% training masks, LoRA versus the full supervised port has higher mean image AUROC0.936 versus0.595 and positive-image Dice0.160 versus0.045, but lower native pixel AP0.200 versus0.254. All three fitted pairs have those respective directions. Against the adapted supervised CNN, the20% Dice difference is much smaller:0.160 versus0.155. At full masks the adapted CNN has higher Dice0.198 versus LoRA0.149. The two CNN implementations therefore must remain distinct; no blanket CNN-superiority or DINO-superiority claim follows.</p>
<p>The full supervised port has zero normal-structure pixel FPR in these fitted source cells while its mean image-level review fraction exceeds96%. Pixel operating-point specificity and image review workload are different endpoints. The additional empty/zero-overlap counts below prevent interpreting low structural FPR as successful crack recovery. No pixel cutoff or source image operating point was changed.</p>
<p>DINO dense systems use a learned global/local image head with image BCE plus pixel CE/Dice. The published architecture port uses pixel supervision and top-local-score image aggregation, with its registered teacher consistency mechanism in SSL. Pretraining, losses, score construction, optimization and additional unlabelled images differ. These results compare implemented pipelines, not an isolated Transformer-versus-CNN or matching-specific cause. Ported source results are not the authors' published experiment or headline score.</p>'''+''.join(sections)+'''
<p>Native AP remains a whole-cohort point for each fit; normal-structure FPR is shown as a raw ratio. All31 paired contrasts/six endpoints/three seeds remain in the primary readout. Source recall here is selected on these same validation labels and is not an independent sensitivity estimate. Actual mask IDs agree within each seed; upstream UCF574 label/HPO exposure is additional, and SSL uses the remaining source-training images.</p>
<p><a href="native_failure_context.json">Bound whole-cohort failure context</a> | <a href="all54_native_failure_records.csv">All54 failure records</a>. Raw timers are not isolated active compute; the paused SSL seed22 record remains unchanged. All69 fits,97 formal efficiency cases,24 ELPV outer/480inner fits and the final source-reviewed single-registry confirmation remain required. The manuscript is not submission-ready.</p>'''
    (directory / 'report.html').write_text(report)
    files = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in sorted(directory.iterdir())]
    save_json(directory / 'report_inputs.json', {'at': now(), 'identity': identity, 'input_bindings': inputs,
                                               'files': files, 'new_inference_bootstrap_or_threshold_fit': False,
                                               'rendered_browser_QA': False, 'numerical_table_content_reviewed': True})
    save_json(SOURCE / 'latest_report.json', {'path': str((directory / 'report_inputs.json').relative_to(ROOT)),
                                           'sha256': sha256(directory / 'report_inputs.json'),
                                           'report': str((directory / 'report.html').relative_to(ROOT))})
    print(str((directory / 'report.html').relative_to(ROOT)))


if __name__ == '__main__':
    main()
