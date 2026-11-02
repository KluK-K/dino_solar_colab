"""All six complete held-source cells: ranking gains versus inspection workload."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from elresearch.common import ROOT, read_json, save_json, sha256, now

BASE = ROOT/'artifacts/publication_20261017'
SYSTEMS = [
    ('dinov3_frozen', 'DINOv3-S frozen', '#a18ac0', 'o'),
    ('dinov3_lora', 'DINOv3-S LoRA', '#703b9c', 'D'),
    ('dinov3_full', 'DINOv3-S full', '#c37839', '^'),
    ('resnet18_full', 'ResNet-18 full', '#798994', 'v'),
    ('resnet50_full', 'ResNet-50 full', '#3d77a4', 'P'),
    ('efficientnet_b0_full', 'EfficientNet-B0 full', '#278778', 's')]


def main():
    path = BASE/'held_source_development_complete_cells.json'; source = read_json(path)
    cells = source['complete_cells']; planned = {(s, b) for s in ['ARTS', 'CFVS', 'CSIR'] for b in [.2, 1.]}
    if {(c['target_acquisition_source'], c['training_fraction']) for c in cells} != planned:
        raise RuntimeError('This figure requires all six predeclared complete held-source cells')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#81959e', 'axes.labelcolor': '#2c4854', 'xtick.color': '#3e5863', 'ytick.color': '#3e5863'})
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.), sharex=True, sharey=True)
    evidence = []
    for row, budget in enumerate([.2, 1.]):
        for column, held in enumerate(['ARTS', 'CFVS', 'CSIR']):
            ax = axes[row, column]
            cell = next(c for c in cells if c['target_acquisition_source'] == held and c['training_fraction'] == budget)
            if {r['family'] for r in cell['models']} != {s[0] for s in SYSTEMS}:
                raise RuntimeError('Do not omit comparator families')
            ax.axhline(.95, color='#8c9da4', lw=1, ls=(0, (3, 3)), zorder=1)
            for family, label, color, marker in SYSTEMS:
                item = next(m for m in cell['models'] if m['family'] == family)
                if sorted(v['seed'] for v in item['seeds']) != [11, 22, 33]:
                    raise RuntimeError('Every family must retain all three seeds')
                for seed in item['seeds']:
                    if sha256(ROOT/seed['run']) != seed['run_sha256']:
                        raise RuntimeError('Underlying completed source run changed')
                    ax.scatter(seed['metrics']['review_rate'], seed['metrics']['recall'],
                        marker=marker, color=color, s=18, alpha=.28, edgecolors='none', zorder=2)
                ax.scatter(item['mean']['review_rate'], item['mean']['recall'], marker=marker,
                    color=color, s=72, linewidth=.6, edgecolors='white', zorder=3)
                evidence.append({'held_source': held, 'training_fraction': budget, 'family': family,
                    'mean_recall': item['mean']['recall'], 'mean_review_rate': item['mean']['review_rate'],
                    'seeds': item['seeds']})
            ax.set_xlim(-.035, 1.035); ax.set_ylim(-.035, 1.035)
            ax.set_xticks([0, .25, .5, .75, 1.]); ax.set_yticks([0, .25, .5, .75, 1.])
            ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
            ax.grid(color='#e8edef', lw=.65); ax.set_axisbelow(True)
            ax.set_title(f"{chr(97+row*3+column)}  {held} · {int(budget*100)}% training labels", loc='left', fontsize=10.5, pad=11)
            ax.text(.025, .035, f"n = {cell['target_images']}; cracks = {cell['target_positive']}",
                transform=ax.transAxes, fontsize=8, color='#637d87')
            if column == 0: ax.set_ylabel('Crack-positive images detected')
            if row == 1: ax.set_xlabel('Images sent for review')
    handles = [Line2D([], [], marker=m, color=c, linestyle='none', label=l, markersize=6) for _, l, c, m in SYSTEMS]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.075, .925), ncol=3,
        columnspacing=2., handletextpad=.5, frameon=False, fontsize=9)
    fig.suptitle('Better ranking does not ensure a portable inspection threshold',
        x=.04, y=.985, ha='left', fontsize=15, color='#173e4d')
    fig.subplots_adjust(left=.09, right=.985, top=.78, bottom=.19, wspace=.14, hspace=.32)
    fig.text(.04, .095, 'Large markers: means of three fitted seeds; faint markers: individual seeds. No seed scores are ensembled.', fontsize=9, color='#486875')
    fig.text(.04, .053, 'Dashed line: 95% source-recall criterion, not a target guarantee. Decisions use frozen source-validation thresholds.', fontsize=8.5, color='#5c7984')
    fig.text(.04, .014, 'Exploratory SCDD development only; validation labels are additional. Review fractions depend on the evaluated cohort, not field prevalence.', fontsize=8, color='#5c7984')
    output = BASE/'figures'; output.mkdir(exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'held_source_inspection_tradeoff.{extension}', dpi=250, facecolor='white')
    plt.close(fig)
    save_json(output/'held_source_inspection_tradeoff_source.json', {'at': now(), 'input_sha256': sha256(path),
        'plot_code_sha256': sha256(__file__), 'records': evidence,
        'scope': 'All six models, three seeds, all six completed held-source cells; no final confirmation or new threshold fitting.'})
    print('Inspection tradeoff: all36 family/cell summaries and108 individual fitted-seed points')


if __name__ == '__main__': main()
