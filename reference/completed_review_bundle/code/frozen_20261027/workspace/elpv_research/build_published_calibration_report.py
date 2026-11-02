"""Vector/HTML source-calibration report from the complete bound metadata readout."""
from pathlib import Path
import hashlib
import html
import json
import shutil
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'published_source_calibration_readout_v1'
SEEDS = [11, 22, 33]
COLORS = {11: '#276C8E', 22: '#B26B1C', 33: '#7864A5'}
INK = '#183342'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p, x): p.write_text(json.dumps(x, indent=2) + '\n')
def label(c):
    s = c['specification']
    rate = '1e-3' if s['learning_rate'] == .001 else '1e-2'
    return ('SSL  lr=' + rate + f"  neg={s['negative_threshold']:.1f}") if s['semi'] else 'FSL  lr=' + rate


def main():
    if not read(BASE / 'inference_efficiency_v1/status.json')['state'].startswith('queued'):
        raise RuntimeError('Do not compete with formal efficiency measurements')
    latest = read(SOURCE / 'latest.json'); result_path = ROOT / latest['path']
    assert sha(result_path) == latest['sha256']
    data = read(result_path)
    assert data['fits'] == 18 and data['inference_views'] == 30
    for bound in data['identity']['input_bindings']:
        assert sha(ROOT / bound['path']) == bound['sha256'], bound['path']
    csv_path = ROOT / latest['csv']['path']; assert sha(csv_path) == latest['csv']['sha256']
    identity = {'result_sha256': sha(result_path), 'builder_sha256': sha(Path(__file__))}
    target = result_path.parent / 'reporting' / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#8DA0A9', 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': '#45616F',
                         'ytick.color': INK, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    metrics = [('image_AUROC', 'Image AUROC'), ('native_pixel_AP', 'Native crack pixel AP'),
               ('positive_Dice', 'Positive-image Dice')]
    selected = data['verified_registered_selection']['selected']
    def chosen(c):
        s = c['specification']; match = next(v for v in selected if v['semi'] == s['semi'])
        return s['learning_rate'] == match['learning_rate'] and (not s['semi'] or s['negative_threshold'] == match['negative_threshold'])
    def save(fig, name):
        for ext in ['png', 'pdf', 'svg']:
            p = target / f'{name}.{ext}'
            if p.exists(): raise RuntimeError('Preserve prior rendered figure; use a new builder identity')
            fig.savefig(p, dpi=220, facecolor='white')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.3), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        ax.axhspan(1.5, 5.55, color='#F2F7F8', zorder=0)
        ax.axhspan(-.55, 1.5, color='#F5F1F8', zorder=0)
        ax.axhline(1.5, color='#C8D1D7', lw=.8)
        for i, c in enumerate(data['candidates']):
            y = 5 - i; values = c['views']['primary'][metric]['values']
            for j, (seed, value) in enumerate(zip(SEEDS, values)):
                ax.scatter(value, y + (j-1)*.13, s=39, color=COLORS[seed], edgecolor='white', lw=.6, zorder=4)
            ax.scatter(c['views']['primary'][metric]['mean'], y, marker='D', s=25, color=INK, zorder=5)
        ax.set_xlim(0, 1.01); ax.set_ylim(-.55, 5.55)
        ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.grid(axis='x', color='#DCE5E9', lw=.7)
        ax.set_axisbelow(True); ax.set_title(title, loc='left', fontsize=12, pad=12)
        ax.set_xlabel('Primary source-validation point')
    axes[0].set_yticks(range(5, -1, -1), [label(c) + ('  *' if chosen(c) else '') for c in data['candidates']])
    axes[0].tick_params(axis='y', length=0, pad=10)
    fig.suptitle('Complete published-method source calibration: 18 fits', x=.02, ha='left', y=.98, fontsize=17, weight='bold')
    fig.text(.02, .915, 'Same 144 reused validation images; primary inference. * Settings selected within each supervision condition.', fontsize=10.3, color='#526D7A')
    handles = [Line2D([], [], marker='o', color='none', markerfacecolor=COLORS[s], markeredgecolor='none', label=f'Seed {s}', markersize=6) for s in SEEDS]
    handles.append(Line2D([], [], marker='D', color='none', markerfacecolor=INK, label='Three-fit mean', markersize=5))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.59, .137), ncol=4, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.235, right=.98, top=.83, bottom=.265, wspace=.10)
    fig.text(.02, .105, 'SSL: 86 training masks + 344 unlabelled training images. FSL: 430 training masks. Both add 144 validation masks and upstream UCF exposure.', fontsize=8.8, color='#526D7A')
    fig.text(.02, .067, 'Checkpoint choice: fixed sampled pixel AP. Candidate choice: mean primary native AP over three seeds. No target outcomes enter selection.', fontsize=8.8, color='#526D7A')
    fig.text(.02, .029, 'Dots retain every fitted cohort point; no confidence intervals or independent superiority test. Lower-budget seeds also vary the mask subset.', fontsize=8.8, color='#526D7A')
    save(fig, 'all18_primary_calibration')

    ssl = [c for c in data['candidates'] if c['specification']['semi']]
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.7), sharey=True)
    labels = []
    for c in ssl:
        labels.extend(label(c).replace('SSL  ', '') + f' | s{s}' for s in SEEDS)
    for ax, (metric, title) in zip(axes, metrics):
        for i, c in enumerate(ssl):
            if i % 2 == 0: ax.axhspan(11-i*3-2.45, 11-i*3+.45, color='#F2F7F8', zorder=0)
            for j, seed in enumerate(SEEDS):
                y = 11-(i*3+j); a = c['views']['primary'][metric]['values'][j]; b = c['views']['ensemble'][metric]['values'][j]
                ax.plot([a, b], [y, y], color=COLORS[seed], alpha=.55, lw=1.15, zorder=2)
                ax.scatter(a, y, color=COLORS[seed], s=32, zorder=4)
                ax.scatter(b, y, facecolors='white', edgecolors=COLORS[seed], marker='s', s=36, lw=1.1, zorder=3)
        ax.set_xlim(0, 1.01); ax.set_ylim(-.6, 11.6)
        ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.grid(axis='x', color='#DCE5E9', lw=.7)
        ax.set_axisbelow(True); ax.set_title(title, loc='left', fontsize=12, pad=12)
        ax.set_xlabel('Saved source-validation point')
    axes[0].set_yticks(range(11, -1, -1), labels); axes[0].tick_params(axis='y', length=0, pad=9)
    fig.suptitle('Teacher 1 and ensemble: the same 12 SSL fits', x=.02, ha='left', y=.98, fontsize=17, weight='bold')
    fig.text(.02, .925, 'Each line joins two declared inference views at one selected checkpoint; it is not an additional training run.', fontsize=10.3, color='#526D7A')
    handles = [Line2D([], [], marker='o', color='none', markerfacecolor=INK, label='Primary: teacher 1', markersize=6),
               Line2D([], [], marker='s', color='none', markerfacecolor='white', markeredgecolor=INK, label='Fixed teacher-logit ensemble', markersize=6)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.61, .095), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.235, right=.98, top=.85, bottom=.20, wspace=.10)
    fig.text(.02, .060, 'Colors identify seeds as in the primary figure. Both views have their own source-calibrated image operating point; pixel cutoff is 0.5.', fontsize=8.8, color='#526D7A')
    fig.text(.02, .027, 'The registered candidate selection uses primary native pixel AP only. No ensemble retuning, independent replication or target comparison is shown.', fontsize=8.8, color='#526D7A')
    save(fig, 'same_checkpoint_SSL_views')

    def fmt(x):
        return 'not estimable' if x['mean'] is None else f"{x['mean']:.3f} +/- {x['sample_SD']:.3f}"
    def table(view):
        body = []
        for c in data['candidates']:
            if view not in c['views']: continue
            vals = c['views'][view]
            body.append('<tr><td>' + html.escape(label(c) + (' *' if chosen(c) else '')) + '</td>' +
                        ''.join('<td>' + fmt(vals[m]) + '</td>' for m in ['image_AUROC', 'native_pixel_AP', 'positive_Dice', 'review_fraction']) + '</tr>')
        return '<table><thead><tr><th>Source candidate</th><th>Image AUROC</th><th>Native pixel AP</th><th>Positive Dice</th><th>Review fraction</th></tr></thead><tbody>' + ''.join(body) + '</tbody></table>'
    report = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete published-method source calibration</title>
<style>body{max-width:1220px;margin:36px auto;padding:0 24px;font:16px/1.55 system-ui;color:#183342;background:#fff}h1{font-size:30px;line-height:1.2}h2{margin-top:34px;font-size:23px}.note{background:#edf4f6;padding:17px 20px;border-left:4px solid #276c8e}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:11px 9px;text-align:left;border-bottom:1px solid #dce5e9}th{background:#edf4f6}img{display:block;width:100%;margin:22px 0}a{color:#246783}.small{font-size:14px;color:#526d7a}</style>
<h1>Complete published-method source calibration</h1><p class="note"><b>18 fits, six candidates, three seeds.</b> These are reused UCF source-validation results, not independent confirmation, a complete matched DINO comparison or a reproduction of the published paper score.</p>
<p>Primary candidate selection was already performed by the registered runner: FSL learning rate 0.001; SSL learning rate 0.001, negative-confidence threshold 0, primary VAT xi 0.1. This report verifies that selection; it does not make a new choice.</p>
<h2>Every primary candidate</h2><p class="small">Mean +/- sample SD of three fitted systems, not confidence intervals. * Source-selected setting within its supervision condition. AP is an original whole-cohort point for each fit.</p>''' + table('primary') + '''
<img src="all18_primary_calibration.png" alt="All eighteen primary source-calibration fits"><p><a href="all18_primary_calibration.pdf">Vector PDF</a> | <a href="all18_primary_calibration.svg">Editable SVG</a></p>
<h2>Declared same-checkpoint SSL ensemble</h2><p>The primary and ensemble views share one fit and one selected checkpoint. The ensemble is not used to select a candidate.</p>''' + table('ensemble') + '''
<img src="same_checkpoint_SSL_views.png" alt="All twelve paired SSL inference views"><p><a href="same_checkpoint_SSL_views.pdf">Vector PDF</a> | <a href="same_checkpoint_SSL_views.svg">Editable SVG</a></p>
<h2>Supervision and endpoint limits</h2><p>SSL fits use 86 training masks and 344 unlabelled source-training images. FSL fits use 430 training masks and no unlabelled pool. Every condition additionally uses 144 validation masks; upstream UCF selection uses the same 574 labelled source identities. These are not matched total-supervision conditions. Partial-budget seeds change the mask subset as well as stochastic fitting.</p>
<p>All image thresholds and temperatures were source-fitted. Source recall therefore is not an independent high-recall guarantee. The 86 crack-negative source images include other defects and are not a verified healthy-production cohort. Normal-structure pixel FPR is <b>not estimable</b> because this source annotation has no eligible crack-negative structure support; null values are retained. Mean negative-image pixel false-positive fraction is a different endpoint.</p>
<p>Raw training/validation timers are preserved. The resumed seed-33 fit includes the long SIGSTOP interval; no active-compute comparison is reported. Complete formal efficiency remains pending.</p>
<p class="small"><a href="../../all18_fits_30_views.csv">Every fit/view and all ten endpoints (CSV)</a> | <a href="../../results.json">Bound numerical result and provenance</a>. No new inference, threshold/temperature fitting, bootstrap or target scores. Missing endpoints are not silently omitted.</p></html>'''
    report_path = target / 'report.html'; report_path.write_text(report)
    assets = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size}
              for p in sorted(target.iterdir()) if p.suffix in {'.png', '.pdf', '.svg', '.html'}]
    receipt = {'at': now(), 'identity': identity, 'result': {'path': str(result_path.relative_to(ROOT)), 'sha256': sha(result_path)},
               'csv': latest['csv'], 'figures_and_report': assets, 'visual_QA_pending': True,
               'new_model_load_or_inference': False, 'independent_confirmation': False}
    write(target / 'figure_inputs.json', receipt)
    write(SOURCE / 'latest_report.json', {'at': now(), 'path': str((target / 'figure_inputs.json').relative_to(ROOT)),
                                        'sha256': sha(target / 'figure_inputs.json'), 'report': str(report_path.relative_to(ROOT))})
    relative = str(report_path.relative_to(SOURCE))
    (SOURCE / 'report.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=' + html.escape(relative, quote=True) + '"><a href="' + html.escape(relative, quote=True) + '">Complete source report</a>')
    print(json.dumps({'report': str(report_path.relative_to(ROOT)), 'figures': str(target.relative_to(ROOT)), 'fits': 18, 'views': 30}, indent=2))


if __name__ == '__main__': main()
