"""Complete saved-output diagnostic figures; no fitting, inference or bootstrap.

Preparation reads declarations only. Rendering requires the original complete
analysis and its successful 6,212-contrast reporting gate. Every eligible record
is retained; no outcome ordering, model, cutoff, population or example choice.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
FINAL = BASE / 'final_confirmation_v1'
REGISTRY_SHA = 'eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
COHORTS = ['scdd_publication_v1', 'ucf_reported_real_confirmation_v1', 'pvelad_restricted_confirmation_v2']
LABELS = ['SCDD original scored pool: 117', 'UCF scored pool: 507', 'PVEL original scored pool: 12,326']
BINS = ['positive_le_0.5pct', 'positive_0.5_to_2pct', 'positive_gt_2pct']
BIN_LABELS = ['Crack area <= 0.5%', '0.5% < crack area <= 2%', 'Crack area > 2%']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')


def write_once(p, d):
    if p.exists():
        assert read(p) == d
    else:
        save(p, d)


def table(p, rows):
    assert rows and not p.exists()
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prepare():
    registry = read(FINAL / 'frozen_registry.json')
    assert sha(FINAL / 'frozen_registry.json') == REGISTRY_SHA
    for b in registry['evaluator_files']:
        assert sha(ROOT / b['path']) == b['sha256']
    plan = {
        'version': 1, 'registry': ref(FINAL / 'frozen_registry.json'),
        'renderer': ref(Path(__file__).absolute()),
        'scope': 'Presentation of every saved calibration record, frozen native-failure subgroup and original population sensitivity; primary comparisons remain unchanged.',
        'figures': [
            {'name': 'calibration_all_scored_pools', 'records': 1683,
             'display': 'Raw versus original source-temperature Brier/NLL/ECE, all jobs in original scored pools. These are full-pool diagnostics, not matching held-source primary strata or primary12292 PVEL calibration.'},
            {'name': 'native_area_strata_all_fits', 'subgroup_records': 1488,
             'display': 'All 496 native-mask jobs x three frozen positive-area bins; source-threshold image recall versus mean positive-image Dice, color is literal zero-overlap fraction. Unsupported strata stay null.'},
            {'name': 'native_overall_positive_negative_all_fits', 'subgroup_records': 992,
             'display': 'All 496 native-mask jobs x all-positive/all-negative strata. Negative-field pixel false-positive fraction is the original secondary metric and is not normal-structure FPR.'},
            {'name': 'PVEL_all_population_sensitivities', 'model_views': 311,
             'display': 'Every fixed primary/full-pool/broader-screen/line-only population readout, with identical saved predictions and no group CI.'},
        ],
        'ordering': 'Original frozen job order; no outcome sorting or omission.',
        'ratios': 'Original raw ratios on [0,1]; NLL uses all-value common axis extent, without clipping.',
        'training_replication': 'Correlated views, shared fitting and deterministic reuse are not independent training or dataset replications.',
        'native_nulls': 'Null values stay in CSV; their absence from a scatter is counted and is never rendered as zero.',
        'zero_overlap': 'Native Dice exactly zero does not necessarily mean an empty predicted mask.',
        'new_training_inference_resampling_fitted_calibration_or_scientific_choice': False,
    }
    out = BASE / 'complete_confirmation_diagnostics_v1' / f'registry_{REGISTRY_SHA[:12]}_renderer_{sha(Path(__file__))[:12]}'
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / 'presentation_plan.json', plan)
    snapshot = out / Path(__file__).name
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
    assert sha(snapshot) == sha(Path(__file__))
    return registry, plan, out


def style(ax):
    ax.set_xlim(-.02, 1.02)
    ax.set_ylim(-.02, 1.02)
    ax.grid(color='#dce4ec', lw=.6)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def finish(fig, out, name, title, note, right=.96):
    import matplotlib.pyplot as plt
    fig.suptitle(title, fontsize=15, weight='bold', y=.98)
    fig.text(.04, .025, note, fontsize=8, color='#33465b', linespacing=1.4)
    fig.subplots_adjust(left=.07, right=right, bottom=.15, top=.86, wspace=.30, hspace=.38)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'{name}.{ext}', dpi=160, facecolor='white')
    plt.close(fig)


def render(registry, plan, out, review_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    analysis_path = FINAL / 'primary_analysis.json'
    analysis = read(analysis_path)
    review = read(review_path)
    assert review['final_analysis_checked'] and review['all6212_declarations_retained']
    assert review['native_failure_subgroup_records'] == 2480
    bindings = read(review_path.parent / 'input_bindings.json')
    assert bindings[str(analysis_path.relative_to(ROOT))] == sha(analysis_path)
    assert analysis['inputs']['registry_id'] == registry['registry_id'] == review['original_registry_id']
    assert [x['declaration'] for x in analysis['primary_comparisons']] == registry['analysis_plan']['primary_comparisons']
    assert len(analysis['jobs']) == 1683
    # Each render run is immutable. Partial/failed runs stay in their directory.
    run = out / ('render_' + datetime.now(timezone.utc).isoformat().replace(':', '-'))
    run.mkdir()
    save(run / 'registration.json', {'at': datetime.now(timezone.utc).isoformat(),
        'plan': ref(out / 'presentation_plan.json'), 'original_analysis': ref(analysis_path),
        'completed_reporting_review': ref(review_path)})
    models = {m['id']: m for m in registry['models']}
    native = []
    calibration = []
    sensitivities = []
    saved_bindings = []
    for index, job in enumerate(analysis['jobs']):
        model = models[job['model_id']]
        common = dict(job_index=index, model_id=model['id'], family=model['family'], seed=model['seed'],
                      architecture=model['architecture'], cohort=job['cohort'], primary_analysis_stratum=job['analysis_stratum'])
        p = ROOT / job['result']['path']
        assert sha(p) == job['result']['sha256']
        result = read(p)
        saved_bindings.append(job['result'])
        assert result['source_frozen_temperature'] == model['source_temperature']
        assert result['source_frozen_threshold'] == model['source_threshold']
        raw, scaled = result['raw_probability_metrics'], result['source_temperature_probability_metrics']
        assert raw['temperature'] == 1 and scaled['temperature'] == model['source_temperature']
        for metric in ['brier', 'negative_log_likelihood', 'ece_10_equal_width_bins']:
            calibration.append(dict(common, metric=metric, raw=raw[metric], source_temperature_scaled=scaled[metric],
                source_temperature=model['source_temperature'], original_scored_pool_images=result['metrics']['n'],
                population='Original whole scored pool; not the matching held-source primary stratum/PVEL12292 primary view'))
        failure = job['secondary_native_failure_analysis']
        if failure is not None:
            for row in failure['subgroups']:
                native.append(dict(common, **row))
        if job['cohort'].startswith('pvelad'):
            base = job['metrics']
            alternatives = {k: v['metrics'] for k, v in job['population_sensitivities'].items()}
            alternatives['line_crack_only'] = job['line_crack_only_sensitivity']['metrics']
            assert set(alternatives) == {'full_v2_pool', 'broader_similarity_screen', 'line_crack_only'}
            for population, actual in alternatives.items():
                for metric, key in [('known_crack_recall', 'recall'), ('author_normal_FPR', None), ('AUROC', 'auc'), ('average_precision', 'average_precision')]:
                    a = base[key] if key else base['fp'] / (base['fp'] + base['tn'])
                    b = actual[key] if key else actual['fp'] / (actual['fp'] + actual['tn'])
                    sensitivities.append(dict(common, population=population, metric=metric,
                        primary_value=a, sensitivity_value=b, primary_images=base['n'], sensitivity_images=actual['n']))
    assert len(calibration) == 1683 * 3 and len(native) == 2480 and len(sensitivities) == 311 * 3 * 4
    table(run / 'all1683_calibration_jobs_three_metrics.csv', calibration)
    table(run / 'all2480_native_failure_subgroups.csv', native)
    table(run / 'all311_PVEL_models_three_sensitivities_four_metrics.csv', sensitivities)
    save(run / 'saved_result_bindings.json', saved_bindings)

    metrics = [('brier', 'Brier score'), ('negative_log_likelihood', 'Negative log-likelihood'), ('ece_10_equal_width_bins', 'ECE: 10 fixed equal-width bins')]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12))
    for i, cohort in enumerate(COHORTS):
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            rr = [r for r in calibration if r['cohort'] == cohort and r['metric'] == metric]
            xx = [r['raw'] for r in rr]; yy = [r['source_temperature_scaled'] for r in rr]
            style(ax)
            if j == 1:
                extent = max(1., max(xx + yy)) * 1.03
                ax.set_xlim(-.01 * extent, extent); ax.set_ylim(-.01 * extent, extent)
            ax.plot(ax.get_xlim(), ax.get_xlim(), color='#8795a5', lw=.7, zorder=0)
            ax.scatter(xx, yy, s=8, alpha=.42, color='#0a6888', linewidths=0)
            ax.set_xlabel('Raw ' + label, fontsize=8); ax.set_ylabel('Source-temperature ' + label, fontsize=8)
            ax.set_title(f'{LABELS[i]} | {len(rr)} records', fontsize=10)
    finish(fig, run, 'calibration_all_scored_pools', 'Source-frozen temperature scaling | every original saved prediction job',
        'Each point is a saved model/view on its original whole scored pool. Lower values indicate lower probability error.\n'
        'Full-pool SCDD includes all 117 and PVEL all 12,326; these are not matching held-source primary strata or the 12,292-image primary PVEL view.\n'
        'Source temperatures are unchanged. Shared training, correlated views and deterministic reuse are not independent replications. No target calibration is fitted.')

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    for i, cohort in enumerate(COHORTS[:2]):
        for j, subgroup in enumerate(BINS):
            ax = axes[i, j]; style(ax)
            rr = [r for r in native if r['cohort'] == cohort and r['subgroup'] == subgroup]
            valid = [r for r in rr if r['images'] > 0]
            ax.scatter([r['source_image_recall'] for r in valid], [r['mean_positive_image_Dice'] for r in valid],
                       c=[r['zero_overlap_fraction'] for r in valid], cmap='viridis', vmin=0, vmax=1, s=11, alpha=.65, linewidths=0)
            ax.set_title(f'{LABELS[i].split(" ")[0]} | {BIN_LABELS[j]}\n{len(valid)} supported / {len(rr)-len(valid)} null fitted records', fontsize=9)
            ax.set_xlabel('Image recall at source threshold', fontsize=9)
            ax.set_ylabel('Mean positive-image native Dice', fontsize=9)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    cax = fig.add_axes([.945, .23, .012, .55])
    fig.colorbar(sm, cax=cax, label='Literal zero-native-overlap fraction')
    finish(fig, run, 'native_area_strata_all_fits', 'Image decisions and native overlap | all frozen positive-area strata',
        'Every 496 native-mask job appears in each applicable frozen bin; absent denominators stay null in the complete 2,480-row table.\n'
        'Zero native overlap means Dice exactly 0 and need not mean an empty predicted mask. Points are descriptive fitted records, without subgroup CIs.\n'
        'The source-fixed area bins and both image/pixel operating points are unchanged. PVEL boxes do not enter mask-Dice plots.', right=.91)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    for i, cohort in enumerate(COHORTS[:2]):
        for j, subgroup in enumerate(['all_positives', 'all_negatives']):
            ax = axes[i, j]; style(ax)
            rr = [r for r in native if r['cohort'] == cohort and r['subgroup'] == subgroup]
            x = 'source_image_recall' if j == 0 else 'source_image_FPR'
            y = 'mean_positive_image_Dice' if j == 0 else 'mean_negative_image_pixel_FP_fraction'
            valid = [r for r in rr if r[x] is not None and r[y] is not None]
            ax.scatter([r[x] for r in valid], [r[y] for r in valid], s=11, alpha=.5, color='#0a6888', linewidths=0)
            ax.set_xlabel('Image recall' if j == 0 else 'Image FPR', fontsize=10)
            ax.set_ylabel('Positive-image native Dice' if j == 0 else 'Negative-field pixel FP fraction', fontsize=10)
            ax.set_title(f'{LABELS[i].split(" ")[0]} | {subgroup.replace("_", " ")} | {len(valid)} records', fontsize=10)
    finish(fig, run, 'native_overall_positive_negative_all_fits', 'Native failure diagnostics | every eligible fitted system',
        'All-positive and all-negative strata are retained for each of 496 native-mask jobs; no model ranking or adjusted inference is performed.\n'
        'Negative-field pixel FP fraction is a mean of per-image ratios over known pixels. It is NOT normal-structure FPR; UCF normal-structure FPR remains null.\n'
        'Image recall/FPR use original source thresholds; native scores use the original fixed pixel rule. No threshold or native endpoint is substituted.')

    pop_order = ['full_v2_pool', 'broader_similarity_screen', 'line_crack_only']
    endpoint_order = ['known_crack_recall', 'author_normal_FPR', 'AUROC', 'average_precision']
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for i, population in enumerate(pop_order):
        for j, endpoint in enumerate(endpoint_order):
            ax = axes[i, j]; style(ax)
            rr = [r for r in sensitivities if r['population'] == population and r['metric'] == endpoint]
            valid = [r for r in rr if r['primary_value'] is not None and r['sensitivity_value'] is not None]
            ax.plot([0, 1], [0, 1], color='#8795a5', lw=.7, zorder=0)
            ax.scatter([r['primary_value'] for r in valid], [r['sensitivity_value'] for r in valid], s=9, alpha=.45, color='#0a6888', linewidths=0)
            ax.set_title(f'{population.replace("_", " ")}\n{endpoint.replace("_", " ")} | {len(valid)} records', fontsize=9)
            ax.set_xlabel('Primary 12,292 population', fontsize=9); ax.set_ylabel('Fixed sensitivity population', fontsize=9)
    finish(fig, run, 'PVEL_all_population_sensitivities', 'PVEL population sensitivities | all 311 declared model/inference views',
        'The same once-scored 12,326 pool supplies all views. Full pool 12,326, primary 12,292 and broader-screen 11,927 are fixed; line-only removes star-only positives.\n'
        'AUROC/AP depend on constructed composition and remain secondary. No verified PVEL capture-family CI or independent population replication is claimed.\n'
        'Every model/view is retained, including adverse directions. Correlated teacher/ensemble views and deterministic reuse remain correlated.')
    outputs = [ref(p) for p in sorted(run.iterdir()) if p.suffix in ['pdf', 'png', 'svg', 'csv']]
    for b in registry['evaluator_files']:
        assert sha(ROOT / b['path']) == b['sha256']
    assert sha(analysis_path) == read(run / 'registration.json')['original_analysis']['sha256']
    save(run / 'verification.json', {
        'at': datetime.now(timezone.utc).isoformat(), 'registration': ref(run / 'registration.json'),
        'calibration_metric_rows': len(calibration), 'native_failure_subgroup_rows': len(native),
        'population_sensitivity_rows': len(sensitivities), 'outputs': outputs,
        'all_records_retained_no_outcome_ordering': True, 'scientific_registry_evaluator_and_analysis_unchanged': True,
        'new_model_inference_training_calibration_fit_or_bootstrap': False,
        'visual_QA_complete': False, 'manuscript_integrated': False, 'submission_ready': False,
    })
    print(json.dumps({'output': str(run), 'figures': 4, 'all_native_strata': len(native), 'calibration_jobs': len(calibration)//3}, indent=2))


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--render', action='store_true')
    p.add_argument('--verified-review', type=Path)
    args = p.parse_args()
    registry, plan, out = prepare()
    if args.render:
        assert args.verified_review and args.verified_review.is_file()
        render(registry, plan, out, args.verified_review)
    else:
        print(json.dumps({'prepared': str(out), 'target_numeric_outputs_read': False, 'new_figures_rendered': False}))


if __name__ == '__main__':
    main()
