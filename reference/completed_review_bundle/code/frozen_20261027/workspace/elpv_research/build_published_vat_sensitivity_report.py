"""One aggregate figure and all-endpoint tables for the complete VAT contrast."""
from pathlib import Path
import datetime
import hashlib
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'published_vat_sensitivity_readout_v1'
SEEDS = [11, 22, 33]
COLORS = ['#276C8E', '#B26B1C', '#7864A5']
INK = '#183342'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p, v): p.write_text(json.dumps(v, indent=2) + '\n')


def main():
    assert read(BASE / 'inference_efficiency_v1/status.json')['state'].startswith('queued')
    latest = read(SOURCE / 'latest.json'); rp = ROOT / latest['path']; assert sha(rp) == latest['sha256']
    data = read(rp); assert data['fits'] == 6 and data['inference_views'] == 12
    for b in data['identity']['input_bindings'] + latest['csvs']:
        assert sha(ROOT / b['path']) == b['sha256'], b['path']
    identity = {'result_sha256': sha(rp), 'builder_sha256': sha(Path(__file__))}
    target = rp.parent / 'reporting' / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    target.mkdir(parents=True, exist_ok=True)
    assert not (target / 'figure_inputs.json').exists(), 'Preserve existing rendered assets'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': '#8DA0A9',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    metrics = [('image_AUROC', 'Image AUROC'), ('native_pixel_AP', 'Native crack pixel AP'),
               ('positive_Dice', 'Positive-image Dice'), ('review_fraction', 'Reviewed-image fraction')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.6), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for i, seed in enumerate(SEEDS):
            values = [next(r for r in data['rows'] if r['condition'] == c and r['seed'] == seed
                           and r['view'] == 'primary')[metric] for c in ['canonical', 'literal']]
            ax.plot([0, 1], values, marker=['o', '^', 's'][i], ms=6, color=COLORS[i], lw=1.5, alpha=.85)
        means = [data['summaries'][c]['primary'][metric]['mean'] for c in ['canonical', 'literal']]
        ax.plot([0, 1], means, marker='D', ms=5, ls='--', lw=1.25, color=INK, zorder=5)
        ax.set_ylim(0, 1.02); ax.set_xlim(-.18, 1.18); ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_xticks([0, 1], ['Canonical FP32\nxi = 0.1', 'Released-caller\nxi = 0.000001'])
        ax.tick_params(axis='x', length=0, pad=9, labelsize=9)
        ax.grid(axis='y', color='#DCE5E9', lw=.75); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11, loc='left', pad=13)
    axes[0].set_ylabel('Saved primary source-validation point')
    fig.suptitle('Registered VAT scale sensitivity: all three seed pairs', x=.025, y=.98,
                 ha='left', fontsize=17, weight='bold')
    fig.text(.025, .912, 'Both settings: learning rate 0.01, negative-confidence threshold 0; primary teacher-1 inference.',
             fontsize=10, color='#526D7A')
    handles = [Line2D([], [], marker=m, color=c, label=f'Seed {s}', ms=6, lw=1.3)
               for s, c, m in zip(SEEDS, COLORS, ['o', '^', 's'])]
    handles.append(Line2D([], [], marker='D', ls='--', color=INK, label='Three-fit mean', ms=5))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .115), ncol=4, frameon=False)
    fig.subplots_adjust(left=.065, right=.985, top=.80, bottom=.30, wspace=.26)
    fig.text(.025, .085, 'Within each seed, source subsets and initial states match; each fit retains its own source-selected checkpoint and image operating point.',
             fontsize=8.4, color='#526D7A')
    fig.text(.025, .052, 'Both use 86 training masks + 344 unlabelled source images + 144 reused validation masks, with additional upstream UCF selection exposure.',
             fontsize=8.4, color='#526D7A')
    fig.text(.025, .019, 'Six fits; 12 correlated primary/ensemble views retained in the CSV. No confidence intervals, independent confirmation or cross-paper ranking.',
             fontsize=8.4, color='#526D7A')
    for ext in ['png', 'pdf', 'svg']:
        p = target / f'source_vat_paired_primary.{ext}'; assert not p.exists()
        fig.savefig(p, dpi=220, facecolor='white')
    plt.close(fig)

    def point(v): return 'not estimable' if v is None else f'{v:+.6f}'
    def cell(v): return 'not estimable' if v['mean'] is None else f"{v['mean']:.6f} +/- {v['sample_SD']:.6f}"
    def table(view):
        rows = []
        for metric in data['plan']['metrics']:
            a = data['summaries']['canonical'][view][metric]; b = data['summaries']['literal'][view][metric]
            delta = data['paired_differences'][view][metric]
            rows.append('<tr><td>' + html.escape(metric) + '</td><td>' + cell(a) + '</td><td>' + cell(b) +
                        '</td>' + ''.join('<td>' + point(v) + '</td>' for v in delta['values']) +
                        '<td>' + cell(delta) + '</td></tr>')
        return '<div class="scroll"><table><thead><tr><th>Endpoint (raw ratio)</th><th>xi0.1 mean +/- SD</th><th>xi1e-6 mean +/- SD</th><th>Delta s11</th><th>Delta s22</th><th>Delta s33</th><th>Delta mean +/- SD</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    text = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete source VAT sensitivity</title>
<style>body{max-width:1280px;margin:36px auto;padding:0 24px;font:16px/1.55 system-ui;color:#183342}h1{font-size:30px;line-height:1.2}h2{font-size:23px;margin-top:34px}.note{background:#edf4f6;padding:17px 20px;border-left:4px solid #276c8e}img{width:100%;display:block;margin:22px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:10px 7px;border-bottom:1px solid #dce5e9;text-align:left;white-space:nowrap}th{background:#edf4f6}a{color:#246783}.small{font-size:14px;color:#526d7a}</style>
<h1>Complete registered source VAT scale sensitivity</h1><p class="note"><b>Both settings, all three seeds, all 12 declared inference views.</b> This is a retrospective description of a registered source-only numerical contrast. It is not independent confirmation, a target-selected xi choice or a reproduction of the authors' paper score.</p>
<p>The literal released-caller setting (xi=0.000001) is compared with the canonical FP32 setting (xi=0.1) at the same learning rate 0.01 and negative-confidence threshold 0. Source subsets and initialization match within each seed. Both use 86 training masks, 344 unlabelled source-training images and 144 reused validation masks. Checkpoint and image-decision calibration reuse those validation labels; upstream UCF calibration exposes 574 source identities.</p>
<img src="source_vat_paired_primary.png" alt="All three source seed pairs for four primary endpoints"><p><a href="source_vat_paired_primary.pdf">Vector PDF</a> | <a href="source_vat_paired_primary.svg">Editable SVG</a></p>
<p>Primary mean native pixel AP is 0.568 at xi0.1 versus0.541 at xi1e-6; the literal-minus-canonical change is adverse for two seeds and favorable for one. Mean image AUROC is 0.832 versus0.850, but only one seed improves. Positive Dice is 0.264 versus0.267, with two favorable and one adverse pair. No setting wins every endpoint and seed. The smaller xi did not cause wholesale training collapse in these three runs; the earlier initial-step FP32 diagnostic cannot establish failure of the authors' trained implementation or other hardware.</p>
<h2>Primary teacher1: every endpoint and fitted pair</h2><p class="small">Differences are literal minus canonical. Every value is a raw ratio, not percentage points. SD describes three fitted systems; it is not a confidence interval. Whole-cohort native pixel AP remains a point per fitted system.</p>''' + table('primary') + '''
<h2>Declared ensemble at the same checkpoints</h2><p>The ensemble is a correlated inference view, not another training replicate or an inference view selected after observing the results. It keeps its own original source-fitted image threshold and temperature.</p>''' + table('ensemble') + '''
<h2>Limits and unchanged choices</h2><p>All six source fits used the same checkpoint rule, but stopped at different rounds. This is a complete training-pipeline sensitivity, not a fixed-checkpoint derivative experiment or isolated explanation of teacher/decoder mechanisms. Across seeds, the mask subset and stochastic training vary together. Source recall is calibration-dependent; the86 crack negatives include other defects and are not a verified healthy-production sample.</p>
<p>Normal-structure pixel FPR remains <b>not estimable</b> in every view. No eligible crack-negative structure annotations support that endpoint. Negative-image mean pixel FPR is a different endpoint. Missing values remain null in JSON/CSV, rather than zero. Raw fit timers remain in the CSV for provenance, not a formal speed comparison.</p>
<p>The original source selection stays unchanged: supervised and SSL learning rate 0.001, selected SSL negative threshold 0 and xi0.1. The matched xi contrast above uses learning rate 0.01 in both arms; comparing the literal setting directly with the selected learning rate 0.001 would mix two parameter changes. Real development, matched DINO comparisons, formal efficiency, the ELPV supplement and one frozen confirmation still must finish.</p>
<p class="small"><a href="../../all6_fits_12_views.csv">All12 fit/view rows</a> | <a href="../../all6_paired_view_differences.csv">All six paired view differences</a> | <a href="../../results.json">Immutable numerical result and source bindings</a>. No new inference, bootstrap, operating-point fitting or model choice.</p></html>'''
    (target / 'report.html').write_text(text)
    assets = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size}
              for p in sorted(target.iterdir()) if p.suffix in {'.png', '.pdf', '.svg', '.html'}]
    receipt = {'at': now(), 'identity': identity, 'result': {'path': latest['path'], 'sha256': latest['sha256']},
               'csvs': latest['csvs'], 'figures_and_report': assets, 'visual_QA_pending': True,
               'new_model_load_or_inference': False, 'independent_confirmation': False}
    write(target / 'figure_inputs.json', receipt)
    write(SOURCE / 'latest_report.json', {'at': now(), 'path': str((target / 'figure_inputs.json').relative_to(ROOT)),
                                        'sha256': sha(target / 'figure_inputs.json'), 'report': str((target / 'report.html').relative_to(ROOT))})
    relative = str((target / 'report.html').relative_to(SOURCE))
    (SOURCE / 'report.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=' + html.escape(relative, quote=True) + '">')
    print(json.dumps({'report': str((target / 'report.html').relative_to(ROOT)), 'figure_folder': str(target.relative_to(ROOT))}, indent=2))


if __name__ == '__main__': main()
