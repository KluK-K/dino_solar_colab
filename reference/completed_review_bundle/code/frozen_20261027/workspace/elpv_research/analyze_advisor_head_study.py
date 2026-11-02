"""Source-validation report for the registered image-only study; reads predictions only."""
import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import numpy as np

from elresearch.common import ROOT, now, read_json, save_json, sha256


def readable(spec):
    encoder = 'DINOv3-S' if spec['encoder'] == 'dinov3_small' else 'ResNet-18'
    adaptation = {'frozen': 'frozen', 'lora': 'LoRA', 'full': 'full tuning'}[spec['adaptation']]
    head = {'linear': 'linear', 'mlp128': 'MLP 128', 'mlp128_32': 'MLP 128 / 32'}[spec['head']]
    return f'{encoder} | {adaptation} | {head}'


def summarize(values):
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values)>1 else None,
            'n': len(values), 'values': values}


def analyze(folder, make_figures=False):
    folder = Path(folder)
    protocol = read_json(folder / 'protocol.json')
    runs = [read_json(p) for p in sorted((folder / 'runs').glob('*/run.json'))]
    grouped = defaultdict(list)
    for run in runs:
        grouped[run['specification']['name']].append(run)
    summaries, raw = [], []
    for spec in protocol['specifications']:
        items = sorted(grouped[spec['name']], key=lambda item: item['seed'])
        item = {'name': spec['name'], 'label': readable(spec), 'specification': spec, 'seeds': [r['seed'] for r in items]}
        if items:
            for key in ['auc', 'average_precision']:
                item[key] = summarize([r['validation_metrics'][key] for r in items])
            item['macro_f1_at_0_5'] = summarize([r['validation_metrics_at_0_5']['macro_f1'] for r in items])
            item['trainable_parameters'] = items[0]['head_parameters'] + items[0]['trainable_encoder_parameters']
            item['head_parameters'] = items[0]['head_parameters']
        summaries.append(item)
        for run in items:
            raw.append({'configuration': spec['name'], 'seed': run['seed'],
                        'image_auroc': run['validation_metrics']['auc'],
                        'image_average_precision': run['validation_metrics']['average_precision'],
                        'macro_f1_at_0_5': run['validation_metrics_at_0_5']['macro_f1'],
                        'balanced_accuracy_at_0_5': run['validation_metrics_at_0_5']['balanced_accuracy'],
                        'recall_at_source_calibrated_threshold': run['validation_metrics']['recall'],
                        'review_rate_at_source_calibrated_threshold': run['validation_metrics']['review_rate'],
                        'head_parameters': run['head_parameters'], 'trainable_encoder_parameters': run['trainable_encoder_parameters'],
                        'best_epoch': run['best_epoch'], 'epochs_completed': run['epochs_completed'],
                        'optimization_seconds': run['optimization_seconds']})
    if raw:
        with (folder / 'individual_results.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    pairs = []
    for adaptation in ['frozen', 'lora', 'full']:
        first = {r['seed']:r for r in grouped[f'dinov3_{adaptation}_linear']}
        second = {r['seed']:r for r in grouped[f'dinov3_{adaptation}_mlp128_32']}
        common = sorted(first.keys() & second.keys())
        if common:
            pairs.append({'adaptation': adaptation, 'seeds': common,
                          'mlp_minus_linear_auc': summarize([second[s]['validation_metrics']['auc']-first[s]['validation_metrics']['auc'] for s in common]),
                          'mlp_minus_linear_ap': summarize([second[s]['validation_metrics']['average_precision']-first[s]['validation_metrics']['average_precision'] for s in common])})
    summary = {'updated_at': now(), 'study_id': protocol['study_id'], 'completed_runs': len(runs),
               'planned_runs': len(protocol['specifications'])*len(protocol['seeds']),
               'scope': 'reused source validation; exploratory; seed SD is not a confidence interval',
               'configurations': summaries, 'paired_head_comparisons': pairs}
    save_json(folder / 'summary.json', summary)
    if make_figures and runs:
        render_figures(folder, protocol, runs, summaries)
        save_json(folder / 'figure_evidence.json', {
            'updated_at': now(), 'renderer_sha256': sha256(__file__),
            'protocol_sha256': sha256(folder/'protocol.json'),
            'predictions': {str(p.relative_to(folder)): sha256(p)
                            for p in sorted((folder/'runs').glob('*/validation_predictions.npz'))},
            'scope': 'reused source validation; mean and sample SD across training seeds',
            'display': {'png_dpi': 600, 'formats': ['png','pdf','svg'],
                        'auroc_axis': [.5,1.01], 'image_ap_axis': [0,1.01],
                        'dots': 'individual seeds', 'diamond': 'mean', 'whiskers': 'seed SD, not CI'}})
    render_html(folder, protocol, summary)
    return summary


def render_figures(folder, protocol, runs, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from publication_figures import publication_style, export_figure
    publication_style()
    figures = folder / 'figures'; figures.mkdir(exist_ok=True)
    usable = [s for s in summaries if 'auc' in s]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    for ax, key, title, limits in zip(axes, ['auc', 'average_precision'],
                                    ['(a) Image AUROC', '(b) Image average precision'], [(.5, 1.01), (0, 1.01)]):
        for index, item in enumerate(usable):
            values = item[key]['values']; average = item[key]['mean']
            color = {'frozen':'#0072b2', 'lora':'#8b5fbc', 'full':'#d55e00'}[item['specification']['adaptation']]
            ax.scatter(values, np.full(len(values), index), s=23, color=color, alpha=.5, zorder=3)
            ax.scatter(average, index, s=62, marker='D', color=color, edgecolors='white', linewidths=.8, zorder=4)
            sd = item[key]['sd']
            if sd is not None:
                ax.errorbar(average, index, xerr=sd, color=color, capsize=3, linewidth=1, zorder=2)
        ax.set(xlim=limits, title=title, xlabel='Score (higher is better)', yticks=range(len(usable)))
        ax.grid(axis='x', color='#e2e8ee', linewidth=.6); ax.set_axisbelow(True)
    axes[0].set_yticklabels([s['label'] for s in usable], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Custom classification heads on pretrained image features', fontsize=13)
    fig.text(.01, .015, 'Dots: individual seeds; diamonds: mean; whiskers: seed SD where available. '
             '144 reused validation images; image-label supervision only.', fontsize=8, color='#52616b')
    fig.tight_layout(rect=(0, .06, 1, .94)); export_figure(fig, figures / 'classification_comparison'); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.9), sharex=True)
    colors = {'linear':'#0072b2', 'mlp128_32':'#8b5fbc'}
    for column, adaptation in enumerate(['frozen', 'lora', 'full']):
        for run in runs:
            spec = run['specification']
            if spec['encoder'] != 'dinov3_small' or spec['adaptation'] != adaptation or spec['head'] not in colors:
                continue
            history = read_json(folder / 'runs' / f"{spec['name']}_seed{run['seed']}" / 'history.json')
            x = [h['epoch'] for h in history]
            label = f"{spec['head']} / seed {run['seed']}"
            for row, metric in [(0, 'loss'), (1, 'validation_auc')]:
                axes[row, column].plot(x, [h[metric] for h in history], color=colors[spec['head']],
                                       alpha=.75, lw=1.1, label=label)
        axes[0, column].set_title({'frozen':'Frozen encoder', 'lora':'LoRA adaptation', 'full':'Full fine-tuning'}[adaptation])
        axes[1, column].set(xlabel='Epoch', ylim=(.5, 1.01))
        for ax in axes[:, column]: ax.grid(color='#e2e8ee', linewidth=.6)
    axes[0, 0].set_ylabel('Training BCE loss'); axes[1, 0].set_ylabel('Validation image AUROC')
    axes[0, 0].plot([], [], color=colors['linear'], label='Linear head')
    axes[0, 0].plot([], [], color=colors['mlp128_32'], label='MLP 128 / 32')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles[-2:], labels[-2:], fontsize=8)
    fig.suptitle('Training behavior under the same image-label protocol', fontsize=13)
    fig.text(.01, .015, 'Individual seed trajectories. Checkpoint selection: highest source-validation AUROC; '
             'maximum 40 epochs; patience 6.', fontsize=8, color='#52616b')
    fig.tight_layout(rect=(0, .05, 1, .94)); export_figure(fig, figures / 'training_curves'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 3.1)); ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
    def node(x, w, text, fill):
        ax.add_patch(FancyBboxPatch((x,.40),w,.32,boxstyle='round,pad=.009,rounding_size=.016',
                                   facecolor=fill,edgecolor='#81939e',linewidth=.8))
        ax.text(x+w/2,.56,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    boxes = [(.015,.12,'EL image\n336 × 336','#edf2f5'),(.185,.19,'DINOv3-S/16\n384-D CLS vector','#e9eef8'),
             (.425,.13,'Per-image\nLayerNorm','#edf2f5'),(.605,.18,'Custom head\n384 → 128 → 32 → 1','#f0eaf7'),
             (.835,.14,'Sigmoid\nCrack probability','#e7f2ef')]
    for x,w,label,fill in boxes: node(x,w,label,fill)
    for a,b in zip(boxes[:-1],boxes[1:]):
        ax.add_patch(FancyArrowPatch((a[0]+a[1]+.01,.56),(b[0]-.015,.56),arrowstyle='-|>',mutation_scale=11,lw=.9,color='#52616b'))
    ax.text(.28,.22,'Encoder: frozen / Q,V LoRA / full fine-tuning',ha='center',fontsize=9,color='#52616b')
    ax.text(.69,.22,'Heads: linear / one hidden layer / two hidden layers',ha='center',fontsize=9,color='#52616b')
    ax.text(.015,.92,'Advisor experiment: learn a task-specific decision from DINOv3 features',fontsize=12,weight='bold')
    ax.text(.015,.04,'Image labels → binary cross-entropy → gradients in the head and enabled encoder parameters. '
            'No kNN, masks, or target-domain fitting.',fontsize=8.5,color='#52616b')
    export_figure(fig, figures / 'method'); plt.close(fig)


def render_html(folder, protocol, summary):
    def fmt(item, key):
        if key not in item: return 'Pending'
        value = item[key]
        return f"{value['mean']:.4f}" + (f" ± {value['sd']:.4f}" if value['sd'] is not None else ' (one seed)')
    rows = ''.join('<tr><td>'+html.escape(s['label'])+'</td><td>'+str(len(s['seeds']))+'/3</td><td>'+fmt(s,'auc')+
                   '</td><td>'+fmt(s,'average_precision')+'</td><td>'+fmt(s,'macro_f1_at_0_5')+'</td><td>'+
                   (f"{s['trainable_parameters']:,}" if 'trainable_parameters' in s else 'Pending')+'</td></tr>'
                   for s in summary['configurations'])
    comparisons = ''.join(f"<li>{html.escape(p['adaptation'])}: MLP − linear AUROC = "
                          f"{p['mlp_minus_linear_auc']['mean']:+.4f}; paired seeds {p['seeds']}.</li>"
                          for p in summary['paired_head_comparisons'])
    figures = ''.join(f'<figure><img src="figures/{name}.png"><figcaption>{label}</figcaption></figure>'
                      for name,label in [('method','Task-specific head and encoder adaptation'),
                                         ('classification_comparison','Image-only comparisons; all dots are source-validation results'),
                                         ('training_curves','Actual saved epoch histories')]
                      if (folder / 'figures' / f'{name}.png').exists())
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>DINOv3 custom classification head study</title><style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#173442;background:#f1f5f8;margin:0}}
main{{max-width:1120px;margin:36px auto;background:white;padding:38px 44px;border-radius:12px}}
h1{{font-size:30px;line-height:1.2}} h2{{font-size:21px;margin-top:32px}} .tag{{color:#6d51a4;font-weight:600}}
.note{{padding:16px 20px;background:#edf4f7;border-left:4px solid #45839e}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid #dce5ea;padding:11px 8px;text-align:left}}th{{background:#edf4f7}}img{{max-width:100%;height:auto}}
figure{{margin:24px 0}}figcaption,small{{color:#617783;font-size:13px}}a{{color:#206c91}}code{{font-size:13px}}
</style><main><p class="tag">ADVISOR FOLLOW-UP · IMAGE-LABEL SUPERVISION</p>
<h1>Does a custom classification head improve DINOv3 crack inspection?</h1>
<p>We isolate the decision network from encoder adaptation: linear versus nonlinear heads, frozen features versus LoRA versus full fine-tuning.</p>
<div class="note"><b>{summary['completed_runs']} / {summary['planned_runs']} completed runs.</b>
<span id="live">The worker updates this report after each completed run.</span><br>
Source split: 430 training / 144 validation images. Three seeds (11, 22, 33). Source test and external domains remain unopened.</div>
<h2>What is trained?</h2><p>DINOv3 final CLS features (384 dimensions), per-image non-affine LayerNorm, and a binary classification head.
Heads are 384→1, 384→128→1, or 384→128→32→1, with GELU between hidden layers and no dropout.
LoRA adapts query/value projections in the last four blocks (rank 8); full tuning updates the encoder.
ResNet-18 controls use 512-dimensional ImageNet features and the same head protocol.</p>
<p>Image-label BCE only; no segmentation loss, normal-reference bank, or kNN. Frozen embeddings are cached; LoRA and full-tuning runs use live images and backpropagate into the encoder.</p>
<h2>Development results</h2><table><thead><tr><th>Configuration</th><th>Seeds</th><th>Image AUROC</th><th>Image AP</th><th>Macro-F1 at 0.5</th><th>Trainable parameters</th></tr></thead><tbody>{rows}</tbody></table>
<p><small>Mean ± sample SD across completed seeds. A single-seed value is explicitly marked. Image AP is not crack pixel AP.
The same validation set selects checkpoints, so these are exploratory development scores.</small></p>
<h2>Paired head comparisons</h2><ul>{comparisons or '<li>Waiting for matched linear/MLP runs.</li>'}</ul>
{figures}<h2>How this relates to the earlier experiments</h2>
<p>The earlier DeepLabv3 and DINOv3 dense networks also received pixel masks. Their scores belong to a different supervision condition and are not ranked in this table.
The Deitsch 2019 ELPV study uses a different dataset and target; its published CNN scores are contextual, not a matched test of superiority.
The current nine configurations represent two encoder families and several adaptation/head choices, not nine independent backbone models.</p>
<h2>Evidence limits</h2><p>UCF file-level real/simulated provenance remains unresolved. Acquisition groups are not confirmed physical-module IDs.
This validation set has been used in earlier development. No publication superiority or cross-domain reliability is established here.
More layers may overfit; negative results remain in the table.</p>
<p><a href="protocol.json">Registered protocol</a> · <a href="summary.json">Summary</a> ·
<a href="individual_results.csv">Per-seed results</a> · <a href="status.json">Worker status</a> · <a href="training.log">Training log</a></p>
<small>Updated {summary['updated_at']}. Model selection: validation AUROC; 40 epochs maximum, early stopping patience 6.</small>
</main><script>fetch('status.json').then(r=>r.json()).then(s=>{{let t=' Worker: '+s.state+'.';
if(s.current) t+=' '+s.current.name+', seed '+s.current.seed+'.';document.getElementById('live').textContent=t;
if(s.current)fetch('runs/'+s.current.name+'_seed'+s.current.seed+'/history.json').then(r=>r.json()).then(h=>{{if(h.length){{let e=h[h.length-1];document.getElementById('live').textContent+=' Epoch '+e.epoch+', validation AUROC '+e.validation_auc.toFixed(4)+'.';}}}}).catch(()=>{{}});
}}).catch(()=>{{}});</script></html>'''
    (folder / 'report.html').write_text(content)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',default='advisor_head_study_20261016')
    args=p.parse_args(); result=analyze(ROOT/'artifacts'/args.study, make_figures=True)
    print(f"{result['completed_runs']}/{result['planned_runs']} complete")
