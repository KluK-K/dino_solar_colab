"""Every complete held-source dense cell at its original image threshold.

Saved integer confusion counts only. No threshold search, model inference,
new interval, pixel-FPR substitution, or final-confirmation data access.
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from elresearch.common import ROOT, read_json, save_json, sha256, digest, now

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'dense_operating_readout_v1'
ARMS = [('deeplabv3plus_full', 'Adapted CNN', '#7c8e95'),
        ('dinov3_dense_frozen', 'DINO frozen', '#31869a'),
        ('dinov3_dense_lora', 'DINO LoRA', '#7955a8'),
        ('dinov3_dense_reference', 'Normal reference', '#b98135')]


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Archive source-development reporting before confirmation')
    aggregate = read_json(BASE / 'dense_development_complete_cells.json')
    cells = [c for c in aggregate['complete_cells'] if c['fold'] != 'internal']
    if not cells:
        print('No complete held-source dense cell yet; no operating result exported')
        return
    bindings = []; rows = []; supports = {}
    for cell in cells:
        path = ROOT / cell['cell_path']
        if sha256(path) != cell['cell_sha256']: raise RuntimeError('Completed cell changed')
        bindings.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
        if {m['family'] for m in cell['models']} != {a[0] for a in ARMS}:
            raise RuntimeError('Show every registered held-source system')
        for model in cell['models']:
            if sorted(f['seed'] for f in model['fits']) != [11, 22, 33]:
                raise RuntimeError('Require all three fitted seeds')
            for fit in model['fits']:
                path = (ROOT / fit['run']).parent / 'target_development.json'
                binding = next(b for b in cell['identity']['bindings'] if b['run'] == fit['run'])
                if sha256(path) != binding['evaluation_sha256']:
                    raise RuntimeError('Original image operating point changed')
                result = read_json(path); m = result['metrics']
                n = m['tp'] + m['fp'] + m['tn'] + m['fn']
                pos = m['tp'] + m['fn']; neg = m['tn'] + m['fp']
                if n != cell['images'] or pos != result['localization']['positive_images'] or min(pos, neg) <= 0:
                    raise RuntimeError('Do not replace missing denominators or alter the cohort')
                recall = m['tp'] / pos; fpr = m['fp'] / neg; flagged = (m['tp'] + m['fp']) / n
                if not np.isclose(recall, m['recall'], atol=1e-12, rtol=0) or not np.isclose(flagged, m['review_rate'], atol=1e-12, rtol=0):
                    raise RuntimeError('Saved rates disagree with original integer counts')
                support = (n, pos, neg)
                if supports.setdefault(cell['fold'], support) != support:
                    raise RuntimeError('Do not compare different cohort support as one source')
                rows.append({'fold': cell['fold'], 'budget': cell['budget'],
                             'family': model['family'], 'seed': fit['seed'],
                             'images': n, 'positive_images': pos, 'negative_images': neg,
                             'tp': m['tp'], 'fp': m['fp'], 'tn': m['tn'], 'fn': m['fn'],
                             'recall': recall, 'negative_image_FPR': fpr, 'flagged_fraction': flagged})
                bindings.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    identity = {'input_bindings': bindings, 'builder_sha256': sha256(__file__),
                'scope': 'All applicable registered held-source cells, four systems and three fitted seeds. Original source-selected image thresholds unchanged.'}
    folder = OUT / digest(identity)[:20]; folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'results.json'
    if not result_path.exists():
        style = {'font.family': 'DejaVu Sans', 'font.size': 7.2,
                 'axes.spines.top': False, 'axes.spines.right': False,
                 'axes.edgecolor': '#9baeb5', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
        with plt.rc_context(matplotlib.rcParamsDefault), plt.rc_context(style):
            fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), sharex=True, sharey=True)
            for row_index, budget in enumerate([.2, 1.]):
                for col, source in enumerate(['ARTS', 'CFVS', 'CSIR']):
                    ax = axes[row_index, col]; fold = 'holdout_' + source
                    selected = [r for r in rows if r['fold'] == fold and r['budget'] == budget]
                    ax.set_xlim(-.035, 1.035); ax.set_ylim(-.035, 1.035)
                    ax.set_xticks([0, .5, 1.]); ax.set_yticks([0, .5, 1.])
                    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                    ax.grid(color='#e7edef', lw=.6); ax.set_axisbelow(True)
                    if not selected:
                        ax.set_facecolor('#f5f7f8')
                        ax.text(.5, .5, 'Pending complete\nsource cell', ha='center', va='center', color='#718892', fontsize=8)
                    else:
                        for family, label, color in ARMS:
                            points = sorted([r for r in selected if r['family'] == family], key=lambda r: r['seed'])
                            if len(points) != 3: raise RuntimeError('Do not present partial fitted-seed cells')
                            for p, marker in zip(points, ['o', '^', 's']):
                                ax.scatter(p['recall'], p['negative_image_FPR'], marker=marker, s=19, color=color, alpha=.38, edgecolors='none', zorder=2)
                            ax.scatter(np.mean([p['recall'] for p in points]), np.mean([p['negative_image_FPR'] for p in points]),
                                       marker='D', s=28, color=color, edgecolors='white', linewidths=.4, label=label, zorder=3)
                    title = f'{source} · {100*budget:g}% train masks'
                    if selected:
                        n, p, _ = supports[fold]; title += f'\n{n} images / {p} positives'
                    ax.set_title(title, fontsize=8, loc='left', pad=7)
                    if col == 0: ax.set_ylabel('Crack-negative image FPR')
                    if row_index == 1: ax.set_xlabel('Crack-image recall')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if not handles:
                handles = [plt.Line2D([], [], marker='D', color=c, ls='', label=l) for _, l, c in ARMS]
                labels = [a[1] for a in ARMS]
            fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.09, .145), ncol=4, frameon=False, fontsize=7, handletextpad=.5, columnspacing=1.4)
            fig.suptitle('Source-frozen image decisions · every complete held-source cell', fontsize=10, color='#284e5b', y=.99)
            fig.subplots_adjust(left=.105, right=.98, top=.88, bottom=.255, wspace=.20, hspace=.40)
            fig.text(.105, .115, 'Faint marks: seeds 11 (circle), 22 (triangle), 33 (square); diamonds: fitted-seed means.', fontsize=6.6, color='#536f79')
            fig.text(.105, .075, 'Higher recall and lower image FPR are favorable. Each point uses its own source-selected threshold;', fontsize=6.6, color='#536f79')
            fig.text(.105, .043, 'these are operating points, not ROC curves or confidence intervals. Negatives may contain other defects.', fontsize=6.6, color='#536f79')
            fig.text(.105, .01, 'Observed development only; additional validation/upstream annotations. Pixel FPR and human/economic costs differ.', fontsize=6.5, color='#536f79')
            for ext in ['png', 'pdf', 'svg']: fig.savefig(folder / f'image_decisions.{ext}', dpi=300, facecolor='white')
            plt.close(fig)
        with (folder / 'every_fitted_seed.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        save_json(result_path, {'at': now(), 'identity': identity, 'complete_held_source_cells': len(cells),
                  'planned_held_source_cells': 6, 'all_fitted_points': rows,
                  'figures': [{'path': str((folder / f'image_decisions.{ext}').relative_to(ROOT)), 'sha256': sha256(folder / f'image_decisions.{ext}')} for ext in ['png', 'pdf', 'svg']],
                  'new_training_or_inference': False, 'final_confirmation_access': False})
    else:
        if read_json(result_path)['identity'] != identity: raise RuntimeError('Preserve prior source-decision snapshot')
        for f in read_json(result_path)['figures']:
            if sha256(ROOT / f['path']) != f['sha256']: raise RuntimeError('Preserve original source-decision figure')
    save_json(OUT / 'latest.json', {'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path),
                                  'builder_sha256': sha256(__file__)})
    print('Source-frozen image decisions:', len(cells), 'complete cells;', len(rows), 'fitted points')


if __name__ == '__main__':
    main()
