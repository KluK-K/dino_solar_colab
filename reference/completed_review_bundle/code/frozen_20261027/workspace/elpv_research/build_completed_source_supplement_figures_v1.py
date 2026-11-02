"""Complete saved ELPV and matched-UCF metrics, retaining null endpoints."""
from pathlib import Path
import hashlib
import json
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
ELPV = BASE / 'complete_ELPV_supplement_readout_v1/268150aa6a1f25e7196e/results.json'
UCF = BASE / 'matched_UCF_source_readout_v1/e7742c7573c4cb2022a9/results.json'
OUT = BASE / 'completed_source_supplement_figures_v1'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def immutable(p, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)


def main():
    elpv, ucf = read(ELPV), read(UCF)
    assert elpv['counts']['outer_fits'] == 24 and ucf['counts']['fits'] == 12
    for data in [elpv, ucf]: assert sha(ROOT / data['plan']['path']) == data['plan']['sha256']
    elpv_metrics = list(elpv['model_summaries'][0]['metrics'])
    ucf_metrics = list(ucf['model_summaries'][0]['metrics'])
    plan = {'version': 1, 'bindings': [ref(Path(__file__).absolute()), ref(ELPV), ref(UCF)],
            'ELPV_metrics': elpv_metrics, 'UCF_metrics': ucf_metrics,
            'ELPV_order': [(c, f) for c in ['logistic', 'linear_svm', 'rbf_svm'] for f in ['hog_stats', 'dinov3_cls']],
            'UCF_order': [x['system'] for x in ucf['model_summaries']],
            'scales': 'Every probability/classification/Dice/IoU metric is shown on0..1. UCF mean negative-field pixel FPR uses0..0.0025 in its original ratio units. Missing normal-structure annotation retains the panel and every model label, with null values explicitly stated.',
            'points': 'All24ELPVouter-fold fits/all8metrics and all12matched-UCFsource fits/all10metrics retained, including12nullstructure endpoints. Black marks are descriptive means, never CIs.',
            'limits': 'ELPV review-label development is not crack localization, CNN comparison or independent confirmation; four folds share training. UCF uses86training+144validation masks and upstream574-label HPO exposure; source recall/review and native localization remain distinct.'}
    OUT.mkdir(exist_ok=True); immutable(OUT / 'plan.json', plan)
    destination = OUT / digest(plan)[:20]
    if destination.exists():
        for b in read(destination / 'outputs.json')['files']: assert sha(ROOT / b['path']) == b['sha256']
        print(json.dumps({'reused': True, 'outputs': ref(destination / 'outputs.json')})); return
    destination.mkdir(); plotted = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), layout='constrained')
    markers = ['o', 's', '^', 'D']; colors = {'hog_stats': '#D55E00', 'dinov3_cls': '#0072B2'}
    labels = ['HOG Log', 'DINO Log', 'HOG Linear', 'DINO Linear', 'HOG RBF', 'DINO RBF']
    for ax, metric in zip(axes.flat, elpv_metrics):
        for x, (classifier, feature) in enumerate(plan['ELPV_order']):
            rows = sorted([p for p in elpv['per_fit'] if (p['classifier'], p['feature']) == (classifier, feature)], key=lambda p: p['outer_fold'])
            values = []
            for point in rows:
                fold = point['outer_fold']; value = point['metrics_at_default_zero'][metric]; values.append(value)
                ax.scatter(x + (fold - 1.5) * .065, value, marker=markers[fold], color=colors[feature], s=33, zorder=3)
                plotted.append({'study': 'ELPV', 'classifier': classifier, 'feature': feature, 'fold': fold, 'metric': metric, 'value': value})
            ax.plot([x - .22, x + .22], [st.mean(values)] * 2, color='black', lw=1.5)
        ax.set_xticks(range(6), labels, rotation=45, ha='right'); ax.set_ylim(0, 1); ax.set_title(metric.replace('_', ' ')); ax.grid(axis='y', color='#ddd', lw=.6)
    handles = [Line2D([], [], marker='o', color=color, linestyle='', label=feature) for feature, color in colors.items()]
    handles += [Line2D([], [], marker=marker, color='#555', linestyle='', label=f'Outer fold {n}') for n, marker in enumerate(markers)]
    handles += [Line2D([], [], color='black', label='Mean across4folds; no CI')]
    fig.legend(handles=handles, loc='outside lower center', ncol=7, frameon=False)
    fig.suptitle('ELPV expert-review-label supplement | All24nested-development outer fits\n1968reused development images;656course-test images remain unscored; folds share training', fontsize=15)
    for extension in ['png', 'pdf', 'svg']: fig.savefig(destination / ('ELPV_all_outer_metrics.' + extension), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(23, 9), layout='constrained')
    seed_style = {11: ('o', '#0072B2', -.11), 22: ('s', '#D55E00', 0), 33: ('^', '#009E73', .11)}
    for ax, metric in zip(axes.flat, ucf_metrics):
        for x, system in enumerate(plan['UCF_order']):
            rows = sorted([p for p in ucf['per_fit'] if p['system'] == system], key=lambda p: p['seed']); values = []
            for point in rows:
                marker, color, offset = seed_style[point['seed']]; value = point['metrics'][metric]
                plotted.append({'study': 'matched_UCF', 'system': system, 'seed': point['seed'], 'metric': metric, 'value': value})
                if value is not None: ax.scatter(x + offset, value, marker=marker, color=color, s=35, zorder=3); values.append(value)
            if values: ax.plot([x - .23, x + .23], [st.mean(values)] * 2, color='black', lw=1.5)
        ax.set_xticks(range(4), ['FSL port', 'Frozen', 'LoRA', 'Reference'], rotation=35, ha='right')
        ax.set_ylim(0, .0025 if metric == 'mean_negative_field_pixel_FPR' else 1)
        title = metric.replace('_', ' ')
        if metric == 'normal_structure_pixel_FPR':
            ax.text(.5, .5, 'Not provided\nAll12values null', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        if metric == 'mean_negative_field_pixel_FPR': title = 'Negative-field pixel FPR\n(mean of image ratios)'; ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        if metric == 'pooled_native_pixel_AP': title = 'Pooled native pixel AP\n(whole-cohort point per fit)'
        if metric == 'false_positive_rate_at_source_threshold': title = 'Image FPR\n(source threshold)'
        if metric == 'recall_at_source_threshold': title = 'Image recall\n(source threshold)'
        if metric == 'review_rate_at_source_threshold': title = 'Image review rate\n(source threshold)'
        ax.set_title(title, fontsize=10); ax.grid(axis='y', color='#ddd', lw=.6)
    handles = [Line2D([], [], marker=v[0], color=v[1], linestyle='', label=f'Fitted seed {seed}') for seed, v in seed_style.items()]
    handles.append(Line2D([], [], color='black', label='Mean of fitted-seed points; no CI'))
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Matched UCF source validation | All12fits and every declared endpoint\n86training+144validation masks per fit; upstream574-label HPO exposure remains additional;58positive validation images', fontsize=15)
    for extension in ['png', 'pdf', 'svg']: fig.savefig(destination / ('matched_UCF_all_metrics.' + extension), dpi=180)
    plt.close(fig)
    assert len(plotted) == 312 and sum(p['value'] is None for p in plotted) == 12
    immutable(destination / 'plotted_points.json', {'points': plotted, 'stored_points': 312, 'null_points_retained': 12, 'visual_QA': False, 'native_article_integration': False})
    immutable(destination / 'outputs.json', {'files': [ref(p) for p in sorted(destination.iterdir()) if p.is_file()]})
    immutable(OUT / 'latest.json', {'plan': ref(OUT / 'plan.json'), 'outputs': ref(destination / 'outputs.json'), 'plotted_points': ref(destination / 'plotted_points.json'), 'figures': 2, 'visual_QA_pending': True})
    print(json.dumps({'folder': str(destination.relative_to(ROOT)), 'figures': 2, 'stored_points': 312, 'null_points_retained': 12}))


if __name__ == '__main__': main()
