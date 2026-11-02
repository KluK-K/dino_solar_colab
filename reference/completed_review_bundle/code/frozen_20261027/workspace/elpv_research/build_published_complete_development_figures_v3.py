"""Plot every complete published-port development cell without new inference.

No intervals, inference, bootstrapping, threshold choice or model selection.
Each figure includes all original systems/views and three saved fitted seeds.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from elresearch.common import ROOT, now, read_json, save_json, sha256

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'published_complete_development_figures_v3'
SYSTEMS = [
    ('deeplabv3plus_full', 'Adapted CNN', '#34495e'),
    ('dinov3_dense_frozen', 'DINO frozen', '#2979b8'),
    ('dinov3_dense_lora', 'DINO LoRA', '#d66b19'),
    ('dinov3_dense_reference', 'DINO normal reference', '#218b58'),
    ('published_deeplab_fsl/primary', 'Full supervised port', '#8b50a2'),
    ('published_psmt_ssl/primary', 'SSL teacher 1', '#bb4670'),
    ('published_psmt_ssl/ensemble', 'SSL ensemble', '#82752d'),
]
ENDPOINTS = [
    ('AUROC', 'Image AUROC', 'Image ranking; does not establish localization'),
    ('pooled_native_pixel_AP', 'Native pooled pixel AP', 'Whole-cohort point estimates; no pixel confidence intervals'),
    ('positive_image_Dice', 'Positive-image Dice', 'Original quantitative masks at the unchanged native cutoff'),
    ('normal_structure_pixel_FPR', 'Normal-structure pixel FPR', 'Raw ratios on the eligible structure subset; not the entire negative field'),
    ('recall_at_source_threshold', 'Recall at source-frozen threshold', 'Achieved sensitivity can differ between systems'),
    ('review_rate_at_source_threshold', 'Review fraction at source-frozen threshold', 'Lower review is not a matched-recall cost comparison'),
]
SEEDS = [11, 22, 33]
MARKERS = ['o', '^', 's']
POPULATIONS = ['source_validation', 'held_acquisition_development']


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def main():
    if not read_json(BASE / 'inference_efficiency_v1/status.json')['state'].startswith('queued'):
        raise PermissionError('Do not render during isolated formal efficiency measurements')
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('This development report must precede final confirmation')
    index_path = BASE / 'published_complete_development_readout_v1/index.json'
    index = read_json(index_path)
    assert len(index['complete_cells']) == index['planned_cells'] == 6 and not index['pending_cells']
    cells, bindings = {}, [ref(Path(__file__).absolute()), ref(ROOT / 'build_published_complete_development_figures.py'), ref(ROOT / 'build_published_complete_development_figures_v2.py'), ref(BASE / 'published_complete_development_figures_v2/plan.json'), ref(BASE / 'published_complete_development_figures_v1/plan.json'), ref(BASE / 'published_complete_development_figures_v1/0323e0962c1f694b6456/visual_verification.json'), ref(index_path)]
    rows = []
    for cell in index['complete_cells']:
        path = ROOT / cell['path']; assert sha256(path) == cell['sha256']
        data = read_json(path)
        review = path.parent / 'numerical_review.json'; reviewed = read_json(review)
        assert reviewed['result'] == cell and reviewed['all_endpoint_summaries_numerically_reviewed'] == data['endpoint_summaries']
        assert sha256(ROOT / cell['outputs']['path']) == cell['outputs']['sha256']
        for value in read_json(ROOT / cell['outputs']['path'])['files']:
            assert sha256(ROOT / value['path']) == value['sha256']
        bindings += [ref(path), ref(review), cell['outputs']]
        key = (data['fold'], data['budget']); assert key not in cells
        cells[key] = data
        for population in POPULATIONS:
            actual = data['systems'][population]
            expected = SYSTEMS if data['budget'] == .2 else SYSTEMS[:5]
            assert set(actual) == {s for s, _, _ in expected}
            for system, label, color in expected:
                fitted = sorted((r for r in data['all_seed_records'] if r['population'] == population and r['system'] == system), key=lambda r: r['seed'])
                assert [r['seed'] for r in fitted] == SEEDS
                for endpoint, title, scope in ENDPOINTS:
                    values = [r[endpoint] for r in fitted]
                    summary = actual[system][endpoint]
                    assert values == summary['values'] and not summary['SD_is_confidence_interval']
                    for seed, value in zip(SEEDS, values):
                        rows.append({'fold': key[0], 'budget': key[1], 'population': population,
                                     'system': system, 'seed': seed, 'endpoint': endpoint, 'value_raw_ratio': value})
    assert len(rows) == 1296
    OUT.mkdir(exist_ok=True)
    plan = {
        'version': 3, 'kind': 'Complete all-six observed-development graphical readout; amended panel-heading layout',
        'bindings': bindings, 'models_and_views_in_fixed_order': SYSTEMS, 'seeds': SEEDS,
        'endpoints': ENDPOINTS, 'matplotlib_version': matplotlib.__version__,
        'Python_version': platform.python_version(),
        'design': 'One figure per endpoint; 4rows (source20%, held20%, source100%, held100%) by3acquisition columns. Every applicable model/view and all3fitted seeds. Shared xscale within endpoint. Black short vertical marks are arithmetic means, not error bars or confidence intervals. Not-registered SSL100% rows explicitly visible.',
        'scales': 'All values, including FPR, remain raw ratios. Five endpoints use0..1. FPR uses0..maximum across all six complete cells, with padding; zero is retained. No favorable panel rescaling.',
        'inference_limits': 'Both UCF/SCDD development informed study design; source validation selected checkpoints, held cohorts already observed. Budgets are reused cohorts, teacher/ensemble correlated views. Three fitted seeds do not yield independent domains, significance or uncertainty intervals. AP is cohort-level point only; no pixel pseudo-replication.',
        'exposure': 'All actual training/validation/SSL counts remain in original complete result rows. Training-budget labels do not include validation labels or additional upstream UCF574 exposure.',
        'source_choices': 'No checkpoint, inference view, threshold, temperature, cutoff, score direction, target normalization/memory or example is selected or changed.',
        'layout_amendment': 'Preserve failed v1 headings. Split cohort labels into three short lines; increase vertical separation. Use ASCII title separators in final export. All plotted raw values, systems/seeds and common endpoint scales unchanged.',
        'new_fitting_model_load_inference_or_bootstrap': False,
        'private_handoff': 'Aggregate plots only; no raw EL pixels, masks, per-image labels, predictions or learned weights.'}
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        assert read_json(plan_path) == plan, 'Bound plotting plan/source changed'
    else:
        save_json(plan_path, plan)
    folder = OUT / hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:20]
    folder.mkdir(exist_ok=True)
    assert not (folder / 'outputs.json').exists(), 'Do not refresh completed figure artifacts'
    save_json(folder / 'figure_inputs.json', {'at': now(), 'plan': ref(plan_path), 'saved_points': len(rows),
        'fitted_population_view_records': 216, 'unique_fits_across_cells': 99, 'plots': len(ENDPOINTS),
        'complete_cells': 6, 'declared_contrasts_retained_in_source_report': 114,
        'endpoint_summaries_retained_in_source_report': 684,
        'input_bindings': bindings, 'independent_confirmation': False, 'new_inference_or_fit': False})
    with (folder / 'all_plotted_points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    outputs = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    row_specs = [(budget, pop) for budget in [.2, 1.] for pop in POPULATIONS]
    for endpoint, title, scope in ENDPOINTS:
        figure, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=True, sharey=True)
        figure.subplots_adjust(left=.18, right=.98, top=.85, bottom=.115, hspace=.67, wspace=.12)
        figure.suptitle(title + ' - all six complete development conditions', fontsize=19, x=.53, y=.974)
        figure.text(.53, .943, scope, ha='center', fontsize=12)
        legend = [Line2D([], [], marker=m, color='none', markerfacecolor='#777777', markeredgecolor='white', markersize=7, label=f'Seed {seed}') for seed, m in zip(SEEDS, MARKERS)]
        legend += [Line2D([], [], color='black', marker='|', linestyle='none', markersize=13, label='Mean (no interval)')]
        figure.legend(handles=legend, loc='upper center', bbox_to_anchor=(.53, .93), ncol=4, frameon=False)
        maximum = max(row['value_raw_ratio'] for row in rows if row['endpoint'] == endpoint and row['value_raw_ratio'] is not None)
        limits = (-.018, 1.018) if endpoint != 'normal_structure_pixel_FPR' else (-max(maximum, 1e-8)*.04, max(maximum, 1e-8)*1.08)
        for row_index, (budget, population) in enumerate(row_specs):
            for col, acquisition in enumerate(['ARTS', 'CFVS', 'CSIR']):
                ax = axes[row_index, col]; data = cells[('holdout_'+acquisition, budget)]
                first = next(r for r in data['all_seed_records'] if r['population'] == population)
                pop_label = 'Source validation' if population == 'source_validation' else 'Observed held development'
                ax.set_title(f'{acquisition} | {int(100*budget)}% training masks\n{pop_label}\n n={first["images"]}, positives={first["positive_images"]}, families={first["provided_capture_families"]}', fontsize=9.8, pad=8)
                for y, (system, label, color) in enumerate(SYSTEMS):
                    if system not in data['systems'][population]:
                        ax.text(.50, y, 'Not registered at 100%', color='#8c8c8c', fontsize=8.5, ha='center', va='center', transform=ax.get_yaxis_transform())
                        continue
                    summary = data['systems'][population][system][endpoint]
                    for delta, marker, value in zip([-.15, 0, .15], MARKERS, summary['values']):
                        if value is None:
                            ax.text(.99, y+delta, 'NE', ha='right', va='center', color=color, fontsize=7.5, transform=ax.get_yaxis_transform())
                        else:
                            ax.scatter(value, y+delta, marker=marker, s=31, color=color, edgecolors='white', linewidths=.45, zorder=3)
                    if summary['mean'] is not None:
                        ax.plot([summary['mean'], summary['mean']], [y-.25, y+.25], color='black', linewidth=1.3, zorder=2)
                ax.set_yticks(range(len(SYSTEMS)), [label for _, label, _ in SYSTEMS])
                ax.set_ylim(len(SYSTEMS)-.5, -.5); ax.set_xlim(*limits)
                ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(axis='y', length=0, labelsize=10)
                ax.tick_params(axis='x', labelbottom=True, labelsize=9)
                if endpoint == 'AUROC':
                    ax.axvline(.5, color='#aaa', linestyle=':', linewidth=.8)
                if endpoint == 'normal_structure_pixel_FPR':
                    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, -3), useMathText=True)
                if row_index == 3:
                    ax.set_xlabel('Raw ratio (not percentage points)' if endpoint == 'normal_structure_pixel_FPR' else 'Saved endpoint value')
        notes = [
            'Every applicable model/view and fitted seed is retained; blank SSL full-budget rows mean not registered, not missing favorable results.',
            '20% / 100% reuse the same held acquisition cohorts; source validation selected checkpoints. UCF574 upstream exposure is additional.',
            'Dots are saved fitted-model points. Short black lines are means, not CIs. No bootstrap, adjusted significance or new inference.',
            'Image ranking, native AP, mask overlap, eligible-structure FPR, recall and review are different endpoints. This is observed development, not final confirmation.',
        ]
        for i, line in enumerate(notes):
            figure.text(.025, .075-i*.017, line, fontsize=9.5, color='#444444', va='center')
        for suffix in ['png', 'pdf', 'svg']:
            path = folder / (endpoint + '.' + suffix)
            figure.savefig(path, dpi=150, facecolor='white')
            outputs.append(ref(path))
        plt.close(figure)
    save_json(folder / 'outputs.json', {'at': now(), 'files': [ref(folder / 'figure_inputs.json'), ref(folder / 'all_plotted_points.csv'), *outputs],
        'PNG_visual_QA': False, 'source_PDF_visual_QA': False, 'native_article_integration': False})
    save_json(OUT / 'latest.json', {'at': now(), 'plan': ref(plan_path), 'folder': str(folder.relative_to(ROOT)),
        'outputs': ref(folder / 'outputs.json'), 'figures': 6, 'all_saved_points': len(rows),
        'visual_QA_pending': True, 'new_fit_inference_or_bootstrap': False, 'independent_confirmation': False})
    print({'folder': str(folder.relative_to(ROOT)), 'figures': 6, 'saved_points': len(rows), 'QA_pending': True})


if __name__ == '__main__':
    main()
