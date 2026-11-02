"""Report-only all-cell view of already saved, paired source head contrasts.

No encoder, inference, fit, resampling, threshold choice or new endpoint.
"""
import csv
import html
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT, read_json, save_json, sha256, digest, now

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'complete_mechanism_summary_v1'
KEYS = ['AUROC', 'AP', 'Dice', 'structure_FPR', 'image_recall', 'image_review_fraction']
PANELS = [('AUROC', 'Image AUROC', 1), ('AP', 'Native pixel AP', 1),
          ('Dice', 'Positive-image Dice', 1), ('structure_FPR', 'Structure FPR (pp)', 100)]
CONTRASTS = [('dinov3_postlora_no_reference', 'dinov3_dense_lora'),
             ('dinov3_postlora_reference', 'dinov3_postlora_no_reference')]


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('Observed-development report cannot change after confirmation')
    state = read_json(BASE / 'inference_efficiency_v1/status.json')['state']
    if not (state.startswith('queued') or state in {'complete', 'failed', 'bounded_stop'}):
        raise RuntimeError('No plotting during formal efficiency measurements')
    index = read_json(BASE / 'mechanistic_complete_cells_v1/index.json')
    status = read_json(BASE / 'mechanistic_controls_v1/status.json')
    if (len(index['complete_cells']) != 10 or index['pending_cells'] or
            status['state'] != 'complete' or status['completed_runs'] != 125):
        raise RuntimeError('Require every original mechanism fit and complete source cell')
    latest = read_json(BASE / 'mechanistic_complete_cells_v1/paired_head_readouts/latest.json')
    paired_path = ROOT / latest['path']; pairs = read_json(paired_path)
    if (sha256(paired_path) != latest['sha256'] or
            pairs['identity']['source_cells'] != index['complete_cells'] or len(pairs['pairs']) != 60):
        raise RuntimeError('Require all60 original descriptive fitted pairs')
    cells = []; rows = []; summaries = []
    for entry in index['complete_cells']:
        path = ROOT / entry['path']; cell = read_json(path)
        qa = read_json(path.parent / 'visual_verification.json')
        if (sha256(path) != entry['sha256'] or qa['result'] != entry or
                not qa['all2_actual_PNGs_inspected']):
            raise RuntimeError('Require actually reviewed original complete-cell figures')
        cells.append(cell)
        for second, first in CONTRASTS:
            selected = [p for p in pairs['pairs'] if p['source_cell'] == entry and
                        p['second_minus_first'] == [second, first]]
            if sorted(p['seed'] for p in selected) != [11, 22, 33]:
                raise RuntimeError('Incomplete paired fitted seeds')
            for pair in selected:
                originals = {r['family']: r for r in cell['all_original_fit_records'] if r['seed'] == pair['seed']}
                a, b = originals[first], originals[second]
                for key in KEYS:
                    if b[key] - a[key] != pair['difference'][key]:
                        raise RuntimeError('Original saved endpoint difference changed')
                rows.append({'fold': cell['fold'], 'budget': cell['budget'], 'seed': pair['seed'],
                             'second': second, 'first': first, **pair['difference'],
                             'training_masks': b['training_masks'], 'validation_masks': b['validation_masks'],
                             'evaluation_images': b['evaluation_images'], 'positive_images': b['positive_images'],
                             'second_empty_positive_predictions': b['positive_images_with_empty_prediction'],
                             'first_empty_positive_predictions': a['positive_images_with_empty_prediction']})
            values = {key: [p['difference'][key] for p in selected] for key in KEYS}
            summaries.append({'fold': cell['fold'], 'budget': cell['budget'], 'second': second, 'first': first,
                              'endpoints': {key: {'mean_difference': statistics.mean(v), 'all3_differences': v,
                                                 'positive': sum(x > 1e-12 for x in v),
                                                 'negative': sum(x < -1e-12 for x in v),
                                                 'numerical_tie': sum(abs(x) <= 1e-12 for x in v)}
                                            for key, v in values.items()}})
    identity = {'builder_sha256': sha256(__file__), 'source_cells': index['complete_cells'],
                'paired_readout': latest, 'endpoints': KEYS,
                'rule': 'All10 complete development cells, both existing contrasts and every3 fitted seed; raw ratios preserved, descriptive only.'}
    folder = OUT / digest(identity)[:20]; result_path = folder / 'results.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['identity'] != identity or result['all60_paired_records'] != rows:
            raise RuntimeError('Preserve original all-cell summary')
        for asset in result['figures'] + [result['csv']]:
            if sha256(ROOT / asset['path']) != asset['sha256']:
                raise RuntimeError('Prior summary asset changed')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        labels = [('Internal' if c['fold'] == 'internal' else 'Held ' + c['fold'].removeprefix('holdout_')) +
                  f" / {100*c['budget']:g}%" for c in cells]
        assets = []
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
            fig, axes = plt.subplots(2, 4, figsize=(10.5, 8.3), sharex='col', sharey=True)
            fig.subplots_adjust(left=.20, right=.98, bottom=.20, top=.85, hspace=.55, wspace=.34)
            colors = ['#2d7585', '#7955a8']
            for i, (second, first) in enumerate(CONTRASTS):
                for j, (key, title, scale) in enumerate(PANELS):
                    ax = axes[i, j]
                    ax.axvline(0, ls='--', color='#71838c', lw=.8)
                    for n, cell in enumerate(cells):
                        rr = sorted((r for r in rows if r['fold'] == cell['fold'] and r['budget'] == cell['budget'] and
                                     r['second'] == second and r['first'] == first), key=lambda r: r['seed'])
                        for off, marker, r in zip([-.12, 0, .12], ['o', '^', 's'], rr):
                            ax.scatter(r[key] * scale, n+off, marker=marker, s=20, color=colors[i], alpha=.5, zorder=3)
                        ax.scatter(statistics.mean(r[key] for r in rr) * scale, n, marker='D', s=24,
                                   color='#193f4d', edgecolor='white', linewidth=.4, zorder=4)
                    for split in [3.5, 5.5, 7.5]: ax.axhline(split, color='#cbd7dc', lw=.7)
                    ax.set_yticks(range(10), labels); ax.set_ylim(9.6, -.6)
                    ax.set_title('Δ ' + title + (' ↓' if key == 'structure_FPR' else ' ↑'), loc='left', fontsize=9, pad=10)
                    ax.grid(axis='x', color='#e6edef', lw=.5); ax.tick_params(axis='y', length=0)
                    ax.tick_params(axis='x', labelbottom=True)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                    for side in ['top', 'right', 'left']: ax.spines[side].set_visible(False)
            for j, (key, _, scale) in enumerate(PANELS):
                limit = max(abs(r[key] * scale) for r in rows) * 1.15
                for i in [0, 1]: axes[i, j].set_xlim(-limit, limit)
            fig.text(.035, .965, 'Fresh-head adaptation: gains depend on source and label budget',
                     fontsize=13, weight='bold', color='#193f4d')
            fig.text(.035, .922, 'Every complete cell and all three fitted seeds; descriptive source/development evidence', fontsize=9)
            fig.text(.20, .884, 'a  Fresh decoder − original LoRA parent', fontsize=10, weight='bold', color=colors[0])
            fig.text(.20, .489, 'b  Normal reference − matched fresh decoder', fontsize=10, weight='bold', color=colors[1])
            notes = ['○ / △ / □: fitted seeds 11 / 22 / 33; ◆: arithmetic mean. No intervals or significance tests.',
                     'Higher AUROC/AP/Dice and lower normal-structure FPR are favorable; FPR changes are percentage points.',
                     'Native pooled AP is an original whole-cohort point per fit. All empty-mask failures remain included.',
                     'Rows reuse four evaluation cohorts, with overlapping development pools; they are not ten independent domains.',
                     'Budgets denote source training masks; source-validation masks and upstream UCF labels are additional.',
                     'Fresh heads add freezing/reinitialization and a second training stage; layer depth and compute are not isolated.']
            for i, note in enumerate(notes): fig.text(.035, .153-i*.023, note, fontsize=7.5, color='#506a74')
            for ext in ['png', 'pdf', 'svg']:
                p = folder / ('all10_paired_head_contrasts.'+ext); fig.savefig(p, dpi=240, facecolor='white')
                assets.append({'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)})
            plt.close(fig)
        csv_path = folder / 'all60_pairs_six_endpoints.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result = {'at': now(), 'identity': identity, 'all60_paired_records': rows, 'all20_cell_contrast_summaries': summaries,
                  'figures': assets, 'csv': {'path': str(csv_path.relative_to(ROOT)), 'sha256': sha256(csv_path)},
                  'sign_count_numerical_tolerance': 1e-12, 'original_differences_changed': False,
                  'new_fit_inference_or_statistical_test': False, 'independent_confirmation': False}
        save_json(result_path, result)
    save_json(OUT / 'latest.json', {'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path)})
    picture = next(a for a in result['figures'] if a['path'].endswith('.png'))
    (OUT / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete fresh-head contrasts</title>'
        '<style>body{max-width:1150px;margin:45px auto;font:17px/1.6 system-ui;color:#193f4d}img{width:100%}</style>'
        '<h1>All complete source head contrasts</h1><p>All ten cells and all fitted pairs. These reused development observations do not establish independent superiority, a layer-depth effect or a matching-specific mechanism.</p>'
        f'<img src="{folder.name}/{html.escape(picture["path"].split("/")[-1])}">'
        f'<p><a href="{folder.name}/results.json">All six endpoints, seed differences and annotation exposure</a> · '
        f'<a href="{folder.name}/all60_pairs_six_endpoints.csv">All60 pairs (CSV)</a></p></html>')
    print('Complete source mechanism summary:10cells,60pairs,360endpoint differences;', result_path.relative_to(ROOT))


if __name__ == '__main__':
    main()
