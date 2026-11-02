"""English pilot report built from saved source-validation predictions only."""
import html
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from .common import ROOT, read_json, save_json, now
from .data import records, load_image, load_mask
from .encoders import reproject
from .heads import NormalBank, InspectionHead, prepare_rows, head_predictions

def build_report():
    runs = [read_json(p) for p in sorted((ROOT / 'artifacts/runs').glob('*/run.json'))]
    runs = [r for r in runs if r['dataset'] == 'ucf' and r['seed'] == 11 and r['budget'] == 1.]
    if not runs:
        raise ValueError('No UCF pilot runs exist')
    fig_dir = ROOT / 'artifacts/figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    def label(r):
        if r['kind'] == 'probe':
            name={'dinov3_small':'DINOv3-S','resnet18':'ResNet-18','dinov3_base':'DINOv3-B'}.get(r['encoder']['name'],r['encoder']['name'])
            return ('HOG + statistics' if r['encoder']['name'] == 'hog_stats' else name + ' ' + r['representation'].upper()) + ' / ' + r['classifier'].replace('_', ' ')
        return {'local': 'DINOv3 local head', 'reference': 'DINOv3 normal-reference head', 'normal_patch_cosine': 'Normal-patch nearest neighbors'}[r['method']]
    table = []
    for r in sorted(runs, key=label):
        m = r['validation_metrics']
        table.append({'method': label(r), 'run_id': r['run_id'], 'source_CV_AUROC': r.get('cv_auc'),
                      'validation_AUROC': m['auc'], 'validation_AP': m['average_precision'],
                      'macro_F1': m['macro_f1'], 'balanced_accuracy': m['balanced_accuracy'],
                      'recall': m['recall'], 'review_rate': m['review_rate'], 'threshold': r['threshold'],
                      'image_labels': True, 'pixel_labels': r['kind'] != 'probe', 'publication_eligible': False})
    frame = pd.DataFrame(table); frame.to_csv(ROOT / 'artifacts/initial_results.csv', index=False)
    frame = frame.sort_values('validation_AUROC')
    fig, ax = plt.subplots(figsize=(11, max(7,len(frame)*.44)))
    colors = ['#b78350' if row.method.startswith('HOG') else '#287b94' if not row.pixel_labels else '#6554a4' for row in frame.itertuples()]
    ax.barh(frame['method'], frame['validation_AUROC'], color=colors)
    for i, value in enumerate(frame['validation_AUROC']):
        ax.text(value+.006, i, f'{value:.3f}', va='center', fontsize=10)
    ax.set_xlim(.5, 1.02); ax.set_xlabel('Source validation AUROC (144 images / 49 acquisition groups)')
    ax.set_title('Initial UCF pilot: one seed, no held-out evaluation')
    fig.tight_layout(); fig.savefig(fig_dir / 'validation_auc.png', dpi=180); plt.close(fig)

    selected = [r for r in runs if r['kind'] == 'head' or
                (r['kind'] == 'probe' and ((r['encoder']['name'] == 'hog_stats' and r['classifier'] == 'rbf_svm') or
                 (r['encoder']['name'] == 'dinov3_small' and r['representation'] == 'cls' and r['classifier'] == 'linear_svm')))]
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in selected:
        with np.load(ROOT / 'artifacts/runs' / r['run_id'] / 'validation_predictions.npz', allow_pickle=False) as pred:
            fpr, tpr, _ = roc_curve(pred['y'], pred['scores'])
        ax.plot(fpr, tpr, label=f"{label(r)} ({r['validation_metrics']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='#bbb')
    ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title='Source validation ROC — exploratory comparison')
    ax.legend(loc='lower right', fontsize=9); fig.tight_layout(); fig.savefig(fig_dir / 'validation_roc.png', dpi=180); plt.close(fig)

    heads = {r['method']: r for r in runs if r['kind'] == 'head'}
    caption = ''
    if {'local', 'reference'} <= set(heads):
        rows = records('ucf', 'validation')
        positive = sorted([r for r in rows if r['target']], key=lambda r: (r['crack_area_fraction'], r['id']))
        negative = sorted([r for r in rows if not r['target']], key=lambda r: r['id'])
        examples = [positive[0], positive[len(positive)//2], negative[0]]
        mapped = {}
        for method, run in heads.items():
            encoder = SimpleNamespace(name=run['encoder']['name'], key=run['encoder_key'], patch_size=16, device=torch.device('cpu'))
            folder = ROOT / 'artifacts/runs' / run['run_id']
            bank = NormalBank.load(folder / 'normal_bank.pt') if method == 'reference' else None
            items = prepare_rows(examples, encoder, bank, method, run['geometry_weight'], 'cpu')
            model = InspectionHead(run['dimensions'], run['config']['hidden_dim'], run['config']['top_fraction'])
            model.load_state_dict(torch.load(folder / 'head.pt', map_location='cpu', weights_only=True))
            _, maps = head_predictions(model, items, 'cpu')
            mapped[method] = [reproject(m.reshape(21,21), item['geometry']) for m,item in zip(maps,items)]
        fig, axes = plt.subplots(3,4,figsize=(13,9), constrained_layout=True)
        for i,row in enumerate(examples):
            axes[i,0].imshow(load_image(row), cmap='gray')
            axes[i,1].imshow(load_mask(row) == 1, cmap='gray', vmin=0, vmax=1)
            for j,method in enumerate(['local','reference'],2):
                image_artist = axes[i,j].imshow(mapped[method][i], cmap='magma', vmin=0, vmax=1)
            axes[i,0].set_ylabel(['Small annotated crack', 'Median crack area', 'No crack annotation'][i])
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
        for ax,title in zip(axes[0], ['EL image','Native crack annotation','Local head response','Normal-reference response']):
            ax.set_title(title)
        fig.colorbar(image_artist, ax=axes[:,2:].ravel().tolist(), shrink=.75, label='Predicted local crack response', ticks=[0,.25,.5,.75,1.])
        fig.suptitle('Development examples: coarse patch responses reprojected to native resolution\nFixed selection by validation crack area; these are not test examples', fontsize=12)
        fig.savefig(fig_dir/'validation_examples.png',dpi=180); plt.close(fig)
        save_json(ROOT/'artifacts/validation_figure_selection.json', {'selection':'smallest positive area, median positive area, lexicographic first negative; source validation only','ids':[r['id'] for r in examples]})
        tiny = load_mask(examples[0]) == 1
        misses = {method: float(mapped[method][0][tiny].max()) for method in mapped}
        save_json(ROOT/'artifacts/small_validation_example.json', {'id':examples[0]['id'], 'maximum_response_on_annotated_crack':misses, 'selection':'smallest annotated crack area in source validation; descriptive example, not a new test'})
        caption = '<img src="figures/validation_examples.png" alt="Validation image, annotation and local responses"><p>White areas denote annotated crack regions. Heatmaps are interpolated patch responses, not evidence of physical understanding or fine-pixel reconstruction. Identical color range [0,1]; all examples come from source validation. The first row exposes a small annotated region with weak responses: good image-level AUROC does not establish reliable tiny-crack localization.</p>'
    delta = heads['reference']['validation_metrics']['auc'] - heads['local']['validation_metrics']['auc'] if len(heads)==2 else None
    text = f'Normal-reference AUROC exceeds the matched local head by {delta:.4f} ({delta*100:.2f} percentage points) in this seed. This small development difference does not establish a robust contribution.' if delta is not None else 'Local-head comparison is pending.'
    columns = ['method','source_CV_AUROC','validation_AUROC','validation_AP','macro_F1','balanced_accuracy','recall','review_rate','pixel_labels']
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Initial DINOv3 crack results</title>
    <style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:24px;background:#f5f8fa;color:#172f42}}h1{{line-height:1.2}}img{{width:100%;background:white;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #cbd5dc}}.note{{background:#fff1cf;padding:18px;border-radius:12px}}</style>
    <h1>DINOv3 solar-cell crack classification: initial local results</h1><p class="note">Provenance-unverified UCF pilot. One seed (11), 100% of the 430-image training partition, 144 source validation images. All 146 source test images, the SCDD external domain and the 656-image course test remain unopened for modeling/evaluation.</p>
    <p>Official frozen DINOv3-S/16 at 336×336, 384-dimensional CLS vectors and 21×21 patch vectors. Four register tokens are excluded. The Mac MPS device performed image encoding. No DINOv2 was downloaded or run.</p>
    <p>Image-only probes fit scaling and tune parameters inside source-training grouped CV. Their source-validation thresholds target 95% recall. Local heads additionally use pixel labels; their epoch selection also uses source validation. Comparisons across supervision levels do not isolate architecture alone.</p>
    <img src="figures/validation_auc.png" alt="Validation AUROC by method"><img src="figures/validation_roc.png" alt="ROC curves">
    <h2>Saved experimental results</h2>{frame[columns].to_html(index=False,float_format=lambda x:f'{x:.4f}',escape=True)}
    <h2>Research interpretation</h2><p>Frozen DINOv3 features substantially improve on the HOG pilot. {html.escape(text)} No result here is a held-out or cross-domain claim, and no comparison to a paper score is justified unless the task, split and supervision match.</p>
    {caption}<h2>Next checks</h2><p>Confirm UCF provenance and original-image associations; expand the grouped cohort; repeat seeds and budgets; evaluate localization and difficult normal structures. LoRA code is implemented and tested on a tiny fixture, but has not yet been trained on research data. Preserve external domains until the confirmatory protocol is fixed.</p>
    <p><a href="initial_results.csv">Result table CSV</a> · Updated {html.escape(now())}</p></html>'''
    (ROOT / 'artifacts/initial_results.html').write_text(report)
    return {'report': str(ROOT/'artifacts/initial_results.html'), 'runs': len(runs), 'reference_minus_local_validation_auc': delta}
