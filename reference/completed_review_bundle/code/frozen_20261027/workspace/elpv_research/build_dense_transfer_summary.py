"""Render existing paired source-development estimates; no fitting or new statistics."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from elresearch.common import ROOT, now, read_json, save_json, sha256


BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'dense_transfer_summary_v1'
METRICS = [
    ('AUROC', 'Image AUROC', 'Higher favors LoRA', 1.),
    ('pooled_native_pixel_AP', 'Native pixel AP', 'Higher favors LoRA', 1.),
    ('positive_image_Dice', 'Positive-image Dice', 'Higher favors LoRA', 1.),
    ('normal_structure_pixel_FPR', 'Normal-structure FPR', 'Lower favors LoRA', 100.),
]


def main():
    index_path = BASE / 'dense_development_uncertainty_v1/index.json'
    index = read_json(index_path)
    cells = []
    bindings = [{'path': str(index_path.relative_to(ROOT)), 'sha256': sha256(index_path)}]
    for item in index['complete_cells']:
        if item['fold'] == 'internal':
            continue
        path = ROOT / item['path']
        if sha256(path) != item['sha256']:
            raise RuntimeError('Existing conditional result changed')
        data = read_json(path)
        if data['confirmation_model_scores_opened']:
            raise RuntimeError('This renderer accepts source-development evidence only')
        pair = next(c for c in data['comparisons']
                    if c['first'] == 'deeplabv3plus_full' and c['second'] == 'dinov3_dense_lora')
        cells.append({'fold': data['fold'], 'budget': data['budget'],
                      'images': data['images'], 'positive_images': data['positive_images'],
                      'provided_capture_families': data['provided_capture_families'],
                      'endpoints': pair['endpoints']})
        bindings.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    expected = {(f'holdout_{s}', b) for s in ['ARTS', 'CFVS', 'CSIR'] for b in [.2, 1.]}
    if len(cells) != 6 or {(c['fold'], c['budget']) for c in cells} != expected:
        raise RuntimeError('All six declared held-source development cells are required')
    cells.sort(key=lambda c: (['holdout_ARTS', 'holdout_CFVS', 'holdout_CSIR'].index(c['fold']), c['budget']))
    identity = {'input_bindings': bindings, 'builder_sha256': sha256(__file__),
                'contrast': 'dinov3_dense_lora minus deeplabv3plus_full',
                'statistics': 'Copy original paired means, all fitted-pair differences and existing intervals verbatim'}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = OUT / key
    result_path = folder / 'results.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['identity'] != identity or any(sha256(ROOT / a['path']) != a['sha256'] for a in result['figures']):
            raise RuntimeError('Prior figure or identity changed')
        print('Reused exact existing six-cell summary', result_path)
        return
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    markers = {11: 'o', 22: '^', 33: 's'}
    color = '#74529d'
    style = dict(plt.rcParamsDefault)
    style.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
                  'axes.titlesize': 8.4, 'xtick.labelsize': 7.4, 'ytick.labelsize': 8,
                  'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.8), sharey=True)
        fig.subplots_adjust(left=.155, right=.985, top=.76, bottom=.36, wspace=.36)
        y = np.asarray([5.6, 4.8, 3.5, 2.7, 1.4, .6])
        for ax, (metric, title, favorable, scale) in zip(axes, METRICS):
            extrema = [0.]
            ax.axvline(0., color='#87959c', lw=.85, ls='--', zorder=0)
            for i, cell in enumerate(cells):
                estimate = cell['endpoints'][metric]['estimate']
                seed_points = estimate['fitted_pairs']
                assert [p['first_seed'] for p in seed_points] == [11, 22, 33]
                point = scale * estimate['mean_pair_difference']
                interval = estimate['acquisition_family_percentile_ci95']
                if interval is not None:
                    lower, upper = [scale * x for x in interval]
                    ax.plot([lower, upper], [y[i], y[i]], lw=1.35, color=color, zorder=2)
                    ax.vlines([lower, upper], y[i]-.065, y[i]+.065, color=color, lw=1.)
                    extrema.extend([lower, upper])
                for offset, pair in zip([-.085, 0., .085], seed_points):
                    value = scale * pair['difference']
                    ax.scatter(value, y[i]+offset, s=15, marker=markers[pair['first_seed']],
                               color=color, alpha=.35, edgecolors='white', linewidths=.3, zorder=3)
                    extrema.append(value)
                    rows.append({'fold': cell['fold'], 'train_budget': cell['budget'], 'endpoint': metric,
                                 'seed': pair['first_seed'], 'raw_difference': pair['difference'],
                                 'display_difference': value, 'display_multiplier': scale,
                                 'raw_mean_difference': estimate['mean_pair_difference'],
                                 'raw_ci_lower': interval[0] if interval else '',
                                 'raw_ci_upper': interval[1] if interval else ''})
                ax.scatter(point, y[i], s=25, marker='D', color=color, edgecolors='white', linewidths=.45, zorder=4)
            low, high = min(extrema), max(extrema)
            width = max(high-low, .02)
            ax.set_xlim(low-.12*width, high+.12*width)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
            ax.set_title(title+'\n'+favorable, loc='left', pad=12, linespacing=1.5)
            ax.set_xlabel('Difference (pp)' if scale == 100. else 'Difference')
            ax.set_ylim(.15, 6.1)
            ax.set_yticks(y)
            ax.set_yticklabels([f"{c['fold'].removeprefix('holdout_')} · {c['budget']:.0%}" for c in cells])
            ax.tick_params(axis='y', length=0, pad=8)
            ax.grid(axis='x', color='#e8edef', lw=.65)
            for yy in [4.15, 2.05]:
                ax.axhline(yy, color='#e8edef', lw=.65)
            for side in ['top', 'right', 'left']:
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color('#9babb1')
        fig.text(.155, .945, 'DINOv3 LoRA versus adapted CNN', fontsize=13, color='#183f4d', weight='bold')
        fig.text(.155, .89, 'Every declared held-source development cell · LoRA minus CNN', fontsize=8.5)
        handles = [Line2D([], [], marker='D', color='none', markerfacecolor=color, markersize=4.5, label='Mean'),
                   *[Line2D([], [], marker=marker, color='none', markerfacecolor=color, alpha=.4,
                            markersize=4, label=f'Seed {seed}') for seed, marker in markers.items()]]
        fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.145, .212), ncol=4,
                   frameon=False, columnspacing=1.2, handletextpad=.45, fontsize=7.4)
        notes = [
            'Train-mask budgets: 20% and 100%. Each source reuses the same evaluation images across budgets.',
            'ARTS: 151 images / 84 provided families; CFVS: 158 / 92; CSIR: 162 / 103. Families are not verified modules.',
            'Bars: existing 95% paired capture-family intervals, conditional on three fitted pairs; no multiplicity adjustment.',
            'Pooled native AP is point-only. Pixels are never bootstrap units. Fixed-cutoff Dice and pixel ranking differ.',
            'Observed development, with extra source-validation and upstream UCF labels; independent confirmation pending.',
        ]
        for i, note in enumerate(notes):
            fig.text(.155, .187-i*.031, note, fontsize=6.75, color='#526c77')
        assets = []
        for ext in ['png', 'pdf', 'svg']:
            p = folder / ('paired_transfer_summary.'+ext)
            fig.savefig(p, dpi=240, facecolor='white')
            assets.append({'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)})
        plt.close(fig)
    csv_path = folder / 'all_paired_values.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {'at': now(), 'identity': identity, 'cells': cells, 'figures': assets,
              'csv': {'path': str(csv_path.relative_to(ROOT)), 'sha256': sha256(csv_path)},
              'all_paired_values': len(rows), 'new_statistics_or_inference': False,
              'final_confirmation_access': False}
    save_json(result_path, result)
    save_json(OUT / 'latest.json', {'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path)})
    print('Rendered all6 held-source cells,24 paired endpoint summaries and72 seed-endpoint points', folder)


if __name__ == '__main__':
    main()
