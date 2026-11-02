"""Complete conditional localization contrasts, with all failures retained."""
import csv
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from elresearch.common import ROOT, digest, now, read_json, save_json, sha256

BASE = ROOT / 'artifacts/publication_20261017/dense_development_uncertainty_v1'
NAMES = {'deeplabv3plus_full': 'CNN', 'dinov3_dense_frozen': 'Frozen',
         'dinov3_dense_lora': 'LoRA', 'dinov3_dense_full': 'Full',
         'dinov3_dense_reference': 'Reference', 'dinov3_reference_no_position': 'No position',
         'dinov3_reference_mean_evidence': 'Mean evidence'}
PLOTTED = ['AUROC', 'positive_image_Dice', 'normal_structure_pixel_FPR']


def render(result, destination):
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
            'pdf.fonttype': 42, 'svg.fonttype': 'none'})
        comparisons = result['comparisons']; n = len(comparisons)
        height = max(5.8, .38*n+2.6)
        fig, axes = plt.subplots(1, 3, figsize=(11.8, height), sharey=True)
        labels = [NAMES[r['second']]+' − '+NAMES[r['first']] for r in comparisons]
        titles = ['Image AUROC difference', 'Positive-image Dice difference', 'Normal-structure FPR difference']
        directions = ['Positive favors second', 'Positive favors second', 'Negative favors second']
        for ax, endpoint, title, direction in zip(axes, PLOTTED, titles, directions):
            factor=100. if endpoint=='normal_structure_pixel_FPR' else 1.
            values = [0.]
            for y, row in enumerate(comparisons):
                item = row['endpoints'][endpoint]
                if item['status'] != 'estimated':
                    ax.text(.98, y, 'Not estimable', transform=ax.get_yaxis_transform(), ha='right', fontsize=7)
                    continue
                estimate = item['estimate']; center = factor*estimate['mean_pair_difference']
                interval = estimate['acquisition_family_percentile_ci95']
                points = [factor*r['difference'] for r in estimate['fitted_pairs']]
                color = '#a46f2d' if row['second'] == 'dinov3_dense_reference' else '#7854a3' if row['second'] == 'dinov3_dense_lora' else '#32778b'
                ax.scatter(points, np.full(len(points), y)+np.linspace(-.09, .09, len(points)), s=13, color=color, alpha=.4)
                if interval is not None:
                    lo, hi = [factor*v for v in interval]
                    ax.hlines(y, lo, hi, color=color, lw=1.3)
                    ax.vlines([lo, hi], y-.07, y+.07, color=color, lw=1.1)
                    values.extend([lo,hi])
                ax.scatter([center], [y], color=color, s=25, zorder=4)
                values.extend(points+[center])
            bound = max(.01 if factor==100 else .025, max(abs(v) for v in values)*1.13)
            ax.set_xlim(-bound, bound); ax.set_ylim(n-.4, -.6)
            ax.set_yticks(range(n), labels); ax.axvline(0, color='#81929a', lw=.8, ls='--', zorder=0)
            ax.set_title(title+'\n'+direction, loc='left', fontsize=9.5, pad=10)
            ax.set_xlabel('Percentage points (second − first)' if factor==100 else 'Second minus first'); ax.grid(axis='x', alpha=.13)
            ax.spines[['top', 'right']].set_visible(False)
        fig.subplots_adjust(left=.20, right=.98, top=.77, bottom=.30, wspace=.16)
        fig.suptitle(f"{result['fold']} · {int(result['budget']*100)}% training masks · conditional paired differences",
            x=.20, y=.96, ha='left', fontsize=12, color='#193e48')
        fig.text(.20, .85, f"{result['images']} images / {result['positive_images']} crack positives / {result['provided_capture_families']} provided capture families",
            fontsize=8.5, color='#536f79')
        fig.text(.20, .14, 'Solid point: mean fitted-pair difference; faint points: all three paired seeds.\n'
            'Bars: percentile intervals from 1,000 requested paired capture-family replicates; conditional on fitted models.',
            fontsize=8, color='#536f79', linespacing=1.5)
        fig.text(.20, .045, 'Exploratory development; no multiplicity-adjusted test. Internal validation selected checkpoints.\n'
            'Capture families are not verified physical modules. Pooled native AP stays point-only in the table; pixels are never bootstrap units.',
            fontsize=8, color='#536f79', linespacing=1.5)
        # Smaller numeric ranges can produce long, crowded default tick labels.
        # Inspect actual label boxes; only reduce tick density, never data limits.
        tick_adjustments=[]
        for axis_index,ax in enumerate(axes):
            for bins in [None,5,4,3]:
                if bins is not None:ax.xaxis.set_major_locator(MaxNLocator(nbins=bins,min_n_ticks=3))
                fig.canvas.draw();renderer=fig.canvas.get_renderer();lo,hi=ax.get_xlim()
                labels=[t.get_window_extent(renderer) for value,t in zip(ax.get_xticks(),ax.get_xticklabels()) if lo<=value<=hi and t.get_visible()]
                overlap=any(a.x1+2>b.x0 for a,b in zip(labels,labels[1:]))
                if not overlap:
                    if bins is not None:tick_adjustments.append({'axis_index':axis_index,'maximum_bins':bins})
                    break
            else:raise RuntimeError('Do not export overlapping axis tick labels')
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(destination/f'paired_differences.{extension}', dpi=230, facecolor='white')
        plt.close(fig)
        return tick_adjustments


def main():
    index = read_json(BASE/'index.json'); pages=[]; rows=[]
    for entry in index['complete_cells']:
        source=ROOT/entry['path']
        if sha256(source)!=entry['sha256']:
            raise RuntimeError('Saved complete conditional result changed')
        result=read_json(source)
        identity={'result_sha256':entry['sha256'],'generator_sha256':sha256(__file__)}
        destination=source.parent/'figures'/digest(identity)[:20]; destination.mkdir(parents=True,exist_ok=True)
        tick_adjustments=render(result,destination)
        save_json(destination/'figure_inputs.json',{'at':now(),'identity':identity,
            'source_result':entry['path'],'endpoints':PLOTTED,'tick_adjustments':tick_adjustments,
            'display':'All applicable declared comparisons; symmetric per-endpoint scales include every interval/seed/mean. FPR multiplied by100 for percentage-point display only; JSON/CSV remain original ratios. No AP interval.'})
        cells=[]
        for contrast in result['comparisons']:
            label=NAMES[contrast['second']]+' − '+NAMES[contrast['first']]
            data=[]
            for endpoint,item in contrast['endpoints'].items():
                estimate=item.get('estimate');interval=estimate['acquisition_family_percentile_ci95'] if estimate else None
                rows.append({'fold':result['fold'],'budget':result['budget'],'first':contrast['first'],
                    'second':contrast['second'],'endpoint':endpoint,'status':item['status'],
                    'difference':estimate['mean_pair_difference'] if estimate else None,
                    'conditional_CI_low':interval[0] if interval is not None else None,
                    'conditional_CI_high':interval[1] if interval is not None else None})
                text='Not estimable' if estimate is None else f"{estimate['mean_pair_difference']:+.4f}"+(
                    f" [{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval is not None else ' (point only / no interval)')
                data.append('<td>'+html.escape(text)+'</td>')
            cells.append('<tr><th>'+html.escape(label)+'</th>'+''.join(data)+'</tr>')
        figure=str((destination/'paired_differences.png').relative_to(BASE))
        headers=''.join('<th>'+html.escape(k)+'</th>' for k in result['comparisons'][0]['endpoints'])
        pages.append(f'<section><h2>{html.escape(result["fold"])} · {int(result["budget"]*100)}% training masks</h2><p>{html.escape(result["population"])}</p><img src="{figure}" alt="All conditional model differences and all fitted seeds"><p><a href="{figure[:-4]}.pdf">Vector PDF</a> · <a href="{str(source.relative_to(BASE))}">All fitted pairs and input hashes</a></p><div class="scroll"><table><tr><th>Second − first</th>{headers}</tr>'+''.join(cells)+'</table></div></section>')
        entry={**entry,'figure_directory':str(destination.relative_to(ROOT))}
        # Store the exact render identity separately from rolling report timestamps.
        save_json(source.parent/'latest_figure.json',entry)
    with (BASE/'all_contrasts.csv').open('w') as output:
        if rows:
            writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (BASE/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Dense development conditional uncertainty</title><style>body{font:16px/1.6 system-ui;color:#193e48;background:#f3f6f7;margin:0}main{max-width:1240px;margin:35px auto;padding:28px;background:white}img{width:100%}section{border-top:1px solid #dce5e7;padding-top:20px;margin-top:25px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;text-align:left;border-bottom:1px solid #dce5e7}a{color:#246d81}</style><main><h1>Matched dense development: conditional uncertainty and tradeoffs</h1><p>Each published cell contains every applicable declared contrast and all three paired fitted seeds. Intervals resample provided capture families, conditional on these fitted models. Internal validation selected checkpoints, so these exploratory intervals cannot establish independent superiority or an adjusted hypothesis test.</p><p>Positive differences favor the second model for AUROC/Dice; negative differences favor the second for normal-structure FPR. Review fraction must be interpreted together with recall. Normal-structure FPR is a ratio of original annotation counts on crack-negative images, with image/family weights. Pooled native AP is a whole-cohort point only; per-image AP is never substituted and individual pixels are never resampled.</p>'''+''.join(pages)+f'<p>{len(index["complete_cells"])} of{index["planned_cells"]} source cells analyzed. <a href="plan.json">Fixed analysis specification</a> · <a href="all_contrasts.csv">Every endpoint and comparison</a>. Final confirmation model scores remain unopened.</p></main></html>')
    print('Exported',len(pages),'complete conditional-dense figures;',len(rows),'endpoint contrasts',flush=True)


if __name__=='__main__':main()
