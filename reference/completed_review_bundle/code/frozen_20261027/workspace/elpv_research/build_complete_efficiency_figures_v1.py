"""All 97 original timing jobs in complete descriptive latency/RSS panels."""
from pathlib import Path
import hashlib
import json
import math
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
READOUT = BASE / 'complete_efficiency_readout_v1/4e1f4c8dcffe5ff1c86f'
OUT = BASE / 'complete_efficiency_figures_v1'
GROUPS = [('image', 'Neural image scoring'), ('dense', 'Dense decoder systems'),
          ('mechanism', 'Representation and reference mechanisms'),
          ('published', 'Full published-method ports'),
          ('representation', 'Resolution, pooling and backbone probes'),
          ('classical', 'Classical image probes')]


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def immutable(p, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)


def main():
    data, review = read(READOUT / 'results.json'), read(READOUT / 'numerical_review.json')
    assert review['checked_fit_conditions'] == 388 and review['checked_primary_passes'] == 1164
    assert sha(READOUT / 'results.json') == review['result']['sha256']
    families = list(dict.fromkeys(x['family'] for x in data['family_summaries']))
    groups = {key: [f for f in families if f.startswith(key + '__')] for key, _ in GROUPS}
    assert sum(map(len, groups.values())) == len(families) == 47
    cases = data['cases']; assert len(cases) == 388
    latency = [x['median_pipeline_ms_per_image'] for x in cases]
    rss = [x['process_lifetime_highwater_RSS_bytes'] / 2**20 for x in cases]
    limits = {'latency': [10**math.floor(math.log10(min(latency))), 10**math.ceil(math.log10(max(latency)))],
              'rss': [2**math.floor(math.log2(min(rss))), 2**math.ceil(math.log2(max(rss)))]}
    plan = {'version': 1, 'kind': 'Complete saved formal cost figures; no new measurement',
            'bindings': [ref(Path(__file__).absolute()), ref(READOUT / 'results.json'),
                         ref(READOUT / 'numerical_review.json'), ref(READOUT / 'all_fit_conditions.csv')],
            'groups': groups, 'conditions': [['scdd_publication_v1', 1], ['scdd_publication_v1', 8], ['ucf', 1], ['ucf', 8]],
            'metrics': ['latency', 'rss'], 'shared_axis_limits': limits,
            'points': 'Every registered fitted seed shown separately in each condition, using median of3 uninstrumented passes for latency. Short black mark is arithmetic mean of fitted-seed points, not a confidence interval. Singleton probe seeds remain singletons.',
            'axes': 'All latency figures share one logarithmic ms/image scale. All RSS figures share one logarithmic MiB scale. Rows preserve registered order within original model-family groups, never sorted by favorable outcomes.',
            'memory': 'RSS is process-lifetime highwater including loading/warmup and earlier conditions; it is not training memory, guaranteed instantaneous peak or a sum of overlapping unified-memory pools.',
            'scope': 'Actual M2 Max32GiB/MPS, already observed351SCDD and430UCFsource images. Different tasks, input sizes and annotation/reference exposure remain explicit in complete tables. No independent accuracy or architecture-only causal comparison.'}
    OUT.mkdir(exist_ok=True); immutable(OUT / 'plan.json', plan)
    destination = OUT / digest(plan)[:20]
    if destination.exists():
        for b in read(destination / 'outputs.json')['files']: assert sha(ROOT / b['path']) == b['sha256']
        print(json.dumps({'reused': True, 'outputs': ref(destination / 'outputs.json')})); return
    destination.mkdir()
    markers = {11: ('o', '#0072B2', -.13), 22: ('s', '#D55E00', 0), 33: ('^', '#009E73', .13)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    panels = []
    for key, title in GROUPS:
        names = groups[key]
        for metric in ['latency', 'rss']:
            fig, axes = plt.subplots(2, 2, figsize=(17, max(8.5, len(names) * .48 + 3.5)), layout='constrained')
            for ax, (cohort, batch) in zip(axes.flat, plan['conditions']):
                subset = [c for c in cases if c['family'] in names and (c['cohort'], c['batch_size']) == (cohort, batch)]
                points = []
                for y, family in enumerate(names):
                    values = []
                    for point in sorted([c for c in subset if c['family'] == family], key=lambda c: c['seed']):
                        marker, color, offset = markers[point['seed']]
                        value = point['median_pipeline_ms_per_image'] if metric == 'latency' else point['process_lifetime_highwater_RSS_bytes'] / 2**20
                        values.append(value); ax.scatter(value, y + offset, marker=marker, color=color, s=32, zorder=3)
                        points.append({'model_id': point['model_id'], 'seed': point['seed'], 'family': family, 'value': value})
                    ax.plot([st.mean(values)] * 2, [y - .23, y + .23], color='black', lw=1.6, zorder=4)
                ax.set_yticks(range(len(names)), [f.split('__', 1)[1] for f in names], fontsize=8)
                ax.invert_yaxis(); ax.set_ylim(len(names) - .55, -.55)
                ax.set_xscale('log', base=10 if metric == 'latency' else 2)
                ax.set_xlim(limits[metric]); ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_xlabel('Pipeline ms/image; median of 3 passes (log scale)' if metric == 'latency' else 'Observed process RSS highwater, MiB (log scale)')
                ax.set_title(('SCDD source inputs (351 images)' if cohort == 'scdd_publication_v1' else 'UCF source inputs (430 images)') + f'; batch {batch}', fontsize=11)
                ax.grid(axis='x', which='both', color='#ddd', linewidth=.6); ax.set_axisbelow(True)
                panels.append({'group': key, 'metric': metric, 'cohort': cohort, 'batch_size': batch, 'points': points})
            handles = [Line2D([], [], marker=v[0], color=v[1], linestyle='', label=f'Fitted seed {seed}') for seed, v in markers.items()]
            handles.append(Line2D([], [], marker='|', markersize=12, color='black', linestyle='', label='Mean of fitted-seed points; no CI'))
            fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
            fig.suptitle(title + (' | End-to-end source inference latency' if metric == 'latency' else ' | Observed RSS highwater') + '\nM2 Max 32 GiB / MPS; all registered systems and fitted seeds retained', fontsize=14)
            stem = destination / (key + '_' + metric)
            for extension in ['png', 'pdf', 'svg']:
                fig.savefig(stem.with_suffix('.' + extension), dpi=180)
            plt.close(fig)
    immutable(destination / 'plotted_points.json', {'panels': panels, 'case_points_per_metric': 388,
               'total_plotted_case_points': sum(len(p['points']) for p in panels),
               'latency_units': 'ms/image', 'RSS_units': 'MiB = 2^20 bytes',
               'PNG_PDF_SVG_visual_QA': False, 'native_article_integration': False})
    immutable(destination / 'outputs.json', {'files': [ref(p) for p in sorted(destination.iterdir()) if p.is_file()]})
    immutable(OUT / 'latest.json', {'plan': ref(OUT / 'plan.json'), 'outputs': ref(destination / 'outputs.json'),
              'plotted_points': ref(destination / 'plotted_points.json'), 'figures_per_format': 12,
              'all_97_jobs_retained': True, 'visual_QA_pending': True})
    print(json.dumps({'folder': str(destination.relative_to(ROOT)), 'figures': 12, 'plotted_points': 776, 'visual_QA_pending': True}))


if __name__ == '__main__': main()
