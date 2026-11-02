"""Export audit and dependency figures from completed non-model evidence only."""
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from elresearch.common import ROOT, read_json, save_json, sha256, now
from elresearch.near_duplicate_audit import image_path, d4
from elresearch.confirmation_population import checked_policy

BASE = ROOT / 'artifacts/publication_20261017'
AUDIT = BASE / 'whole_frame_similarity_audit_v1'
FIG = BASE / 'figures'


def export(figure, name):
    for extension in ['png', 'pdf', 'svg']:
        figure.savefig(FIG / f'{name}.{extension}', dpi=220, facecolor='white')
    plt.close(figure)


def main():
    policy_path = BASE / 'confirmation_preparation_v1/cohort_analysis_policy_v1.json'
    policy = checked_policy({'path': str(policy_path.relative_to(ROOT)), 'sha256': sha256(policy_path)})
    screen = read_json(AUDIT / 'result.json')
    review = read_json(AUDIT / 'origin_review.json')
    measurements = read_json(AUDIT / 'native_resolution_measurements.json')
    groups = read_json(BASE / 'grouping_sensitivity_v1/results.json')
    assert screen['search_complete'] and groups['status'] == 'complete'
    assert not review['model_scores_used']
    records = {r['id']: r for r in read_json(AUDIT / 'input_records.json')['records']}
    # Examples illustrate two observed residual regimes. Neither was chosen
    # using a model response. The complete 17-pair gallery stays accessible.
    chosen = [('pvelad/img000374', 'pvelad/img001223'),
              ('pvelad/img000857', 'pvelad/img000896')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(11.5, 7.6))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.18],
                           left=.035, right=.98, top=.88, bottom=.19,
                           wspace=.18, hspace=.35)
    for row, (first_id, second_id) in enumerate(chosen):
        pair = next(r for r in screen['strong_similarity_pairs']
                    if r['first_id'] == first_id and r['second_id'] == second_id)
        metric = next(r for r in measurements['pairs']
                      if r['first_id'] == first_id and r['second_id'] == second_id)
        a, b = records[first_id], records[second_id]
        with Image.open(image_path(a)) as image:
            first = np.asarray(image.convert('L'), float)
        with Image.open(image_path(b)) as image:
            second = list(d4(np.asarray(image.convert('L')).copy()))[pair['orientation']].astype(float)
        assert first.shape == second.shape
        residual = first - (metric['positive_affine_slope'] * second + metric['affine_intercept'])
        for column, values in enumerate([first, second, residual]):
            ax = fig.add_subplot(grid[row, column])
            if column < 2:
                ax.imshow(values, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                record = [a, b][column]
                h, w = values.shape
                for box in record.get('boxes', []):
                    if box['class'] not in {'crack', 'star_crack'}:
                        continue
                    x1, y1, x2, y2 = box['normalized_xyxy']
                    ax.add_patch(plt.Rectangle((x1*w, y1*h), (x2-x1)*w, (y2-y1)*h,
                                              fill=False, ec='#ffcc55', lw=1.5))
                ax.set_title(record['id'].split('/')[-1], fontsize=10, loc='left')
            else:
                im = ax.imshow(residual, cmap='RdBu_r', vmin=-8, vmax=8, interpolation='nearest')
                ax.set_title('Affine-adjusted difference', fontsize=10, loc='left')
                if row == 1:
                    colorbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                                           fraction=.07, pad=.03, ticks=[-8, 0, 8])
                    colorbar.set_label('Gray-value residual (fixed scale)', fontsize=8)
                    colorbar.ax.tick_params(labelsize=8)
            ax.set_axis_off()
        ax = fig.add_subplot(grid[row, 3]); ax.set_axis_off()
        ax.text(0, .97, ['a  Noise-like residual', 'b  Almost identical pixels'][row],
                transform=ax.transAxes, va='top', weight='bold', color='#183d48')
        description = (f"Native NCC: {metric['correlation_native']:.6f}\n"
                       f"Residual RMSE: {metric['affine_residual_RMSE_gray_units']:.3f}\n\n"
                       'Different crack-box annotations\nBoth images quarantined\nfrom primary analysis\n\n'
                       'Origin is unverified; similarity\ndoes not establish generation\nor physical-module identity.')
        ax.text(0, .78, description, transform=ax.transAxes, va='top',
                fontsize=9, linespacing=1.6, color='#405e68')
    fig.text(.035, .96, 'Pre-inference image-integrity review', fontsize=17,
             weight='bold', color='#183d48', va='top')
    fig.text(.035, .915, 'Native EL pixels and author crack boxes; no model output', fontsize=11, color='#506d77')
    fig.text(.035, .065,
             '17 reviewed PVEL near-copy pairs (34 images); all have differing crack boxes. Every pair is retained in the full audit.\n'
             'Primary: 943 known positives + 11,349 author-normal images. The unchanged 12,326-image pool is scored once for all views.',
             fontsize=9, color='#506d77', linespacing=1.5)
    export(fig, 'preinference_near_copy_evidence')

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True)
    names = {'dinov3_frozen': 'DINOv3 frozen', 'resnet18_full': 'ResNet-18',
             'resnet50_full': 'ResNet-50', 'efficientnet_b0_full': 'EfficientNet-B0'}
    comparator_order = list(names)
    for ax, budget in zip(axes, [.2, 1.]):
        ax.axvline(0, color='#98a9ad', lw=1, ls='--', zorder=0)
        for y, family in enumerate(comparator_order):
            row = next(r for r in groups['comparisons'] if r['training_fraction'] == budget and r['first'] == family)
            for key, offset, color, marker in [('original', -.12, '#7c959d', 'o'),
                    ('merged_possible_related_families', .12, '#236d7f', 's')]:
                values = row[key]['metrics']['auroc']
                point = values['mean_seed_difference']
                low, high = values['acquisition_family_percentile_ci95']
                label = ('Provided families' if key == 'original' else 'Two possible relatives merged') if y == 0 else None
                ax.errorbar(point, y+offset, xerr=[[point-low], [high-point]],
                            fmt=marker, color=color, ms=4, capsize=3, lw=1.2, label=label)
        ax.set_xlim(-.08, .50); ax.set_ylim(3.6, -.65)
        ax.set_title(f'{int(budget*100)}% source-training labels', loc='left', fontsize=11)
        ax.set_xlabel('LoRA minus comparator AUROC')
        ax.set_yticks(range(4), [names[f] for f in comparator_order])
        ax.grid(axis='x', alpha=.15)
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc='lower left',
               bbox_to_anchor=(.17, .15), ncol=2, borderaxespad=0)
    fig.subplots_adjust(left=.17, right=.98, top=.83, bottom=.32, wspace=.12)
    fig.suptitle('Capture-family sensitivity on held-out ARTS development images',
                 x=.17, ha='left', fontsize=13, color='#183d48')
    fig.text(.17, .015,
             'Same saved predictions, 3 fitted seeds and 1,000 paired resamples; point differences stay identical.\n'
             'Intervals condition on these fitted models. The merge is a dependency sensitivity, not verified module identity.',
             fontsize=8.5, color='#506d77', linespacing=1.5)
    export(fig, 'capture_family_sensitivity')
    evidence = {'at': now(), 'model_predictions_used_for_origin_decisions': False,
                'screen_result_sha256': sha256(AUDIT/'result.json'),
                'native_measurements_sha256': sha256(AUDIT/'native_resolution_measurements.json'),
                'origin_review_sha256': sha256(AUDIT/'origin_review.json'),
                'policy_sha256': sha256(policy_path),
                'grouping_sensitivity_sha256': sha256(BASE/'grouping_sensitivity_v1/results.json'),
                'figure_generator_sha256': sha256(__file__),
                'illustrated_pairs': chosen,
                'all_grouping_contrasts': groups['comparisons'],
                'primary_PVEL_counts': {k:v for k,v in policy['cohorts']['pvelad_restricted_confirmation_v2']['subsets']['primary'].items() if k!='excluded_ids'}}
    save_json(FIG/'data_quality_evidence_source.json', evidence)
    links = ''.join(f'<li><a href="native_candidates_page{i}.png">Complete native review, page {i}</a></li>' for i in range(1,5))
    (AUDIT/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Pre-inference data-quality evidence</title><style>body{font:16px/1.6 system-ui;color:#193d48;background:#f3f6f7;margin:0}main{max-width:1100px;margin:40px auto;padding:24px;background:white}img{width:100%}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4e7}a{color:#246b7e}.small{font-size:13px;color:#54707a}</style><main>
<h1>Image integrity before model confirmation</h1>
<p>The audit used nonlearned whole-frame fingerprints and native-pixel review across 15,963 images. Of 4,830 coarse candidates, 238 passed the similarity screen and 18 passed the stronger screen: 17 PVEL pairs and one SCDD source-training pair. A shared origin remains unverified. Three weaker source/confirmation candidates have different printed module identifiers and were retained.</p>
<img src="../figures/preinference_near_copy_evidence.png" alt="Two native PVEL near-copy examples with different author boxes and fixed-scale residuals">
<p class="small">These are data and annotation diagnostics, not prediction maps. Full images were reviewed for integrity; confirmation model scores remain unopened. No image was removed using a model outcome.</p>
<h2>One scoring pool, three fixed analysis views</h2><table><tr><th>View</th><th>Images</th><th>Known crack positive</th><th>Author normal</th></tr>
<tr><td>Primary, reviewed near-copy members excluded</td><td>12,292</td><td>943</td><td>11,349</td></tr>
<tr><td>Original authorized pool, sensitivity</td><td>12,326</td><td>977</td><td>11,349</td></tr>
<tr><td>All screen members excluded, sensitivity</td><td>11,927</td><td>933</td><td>10,994</td></tr></table>
<p>The unmodified master pool is scored once. Summaries use the same saved predictions and predeclared views; these are not independent replications. Broader screening can remove legitimate repeated patterns and is a sensitivity only. Crop overlap, all possible near-copies and pretraining membership are not ruled out. Author-normal does not establish specificity against every other defect.</p>
<h2>Source dependency sensitivity</h2><img src="../figures/capture_family_sensitivity.png" alt="Original and merged capture-family intervals for all eight held ARTS comparisons">
<p>The original source models and splits remain unchanged. Treating ARTS_00020 and ARTS_00081 as potentially related changes interval endpoints but not predictions or point differences. At 20% labels, the LoRA minus EfficientNet interval still spans zero. The source pair does not establish a shared physical module.</p>
<ul>'''+links+'''</ul><p><a href="origin_review.json">Origin review</a> · <a href="result.json">Complete screen</a> · <a href="../confirmation_preparation_v1/cohort_analysis_policy_v1.json">Frozen population policy</a> · <a href="../figures/data_quality_evidence_source.json">Figure evidence and all sensitivity contrasts</a></p>
<p class="small">All figures export as PNG, vector PDF and SVG under the figures directory. This audit does not demonstrate predictive accuracy or publication readiness.</p></main></html>''')
    print('Exported complete data-quality evidence; no new model inference')


if __name__ == '__main__':
    main()
