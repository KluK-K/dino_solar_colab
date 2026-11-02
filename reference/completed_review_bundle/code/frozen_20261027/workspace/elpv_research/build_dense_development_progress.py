"""Export only complete registered dense cells, with every model and seed.

Cell payloads are immutable by input digest. The aggregate is a development
index; it cannot authorize confirmation access or select a favorable model.
"""
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch.metrics import classification

BASE=ROOT/'artifacts/publication_20261017'
SOURCE=BASE/'dense_development_v1'
FOLDER=BASE/'dense_complete_cells_v1'
NAMES={'deeplabv3plus_full':'Adapted DeepLabv3+ / R50','dinov3_dense_frozen':'DINOv3 frozen',
    'dinov3_dense_lora':'DINOv3 LoRA','dinov3_dense_full':'DINOv3 full',
    'dinov3_dense_reference':'DINOv3 + normal reference',
    'dinov3_reference_no_position':'Reference, no position','dinov3_reference_mean_evidence':'Reference, mean evidence'}


def _build_complete_cells():
    if (BASE/'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Archive source-development reporting before real confirmation scoring')
    settings=read_json(SOURCE/'protocol.json');selection_path=SOURCE/'selected_hyperparameters.json'
    if not selection_path.exists():print('Dense source calibration pending');return
    selection=read_json(selection_path);FOLDER.mkdir(exist_ok=True)
    for key in ['scientific_hashes','extra_scientific_hashes']:
        if any(sha256(ROOT/'elresearch'/name)!=value for name,value in settings.get(key,{}).items()):
            raise RuntimeError('Dense scientific source changed')
    complete=[];pending=[]
    for fold in settings['split_manifest']['folds']:
        for budget in fold['budgets']:
            internal=fold['name']=='internal'
            families=[s['name'] for s in selection['selected'] if internal or s['name']!='dinov3_dense_full']+['dinov3_dense_reference']
            if internal and budget in [.2,1.]:families+=['dinov3_reference_no_position','dinov3_reference_mean_evidence']
            expected={(family,seed) for family in families for seed in settings['seeds']}
            found={}
            for path in (SOURCE/'runs'/fold['name']/f'budget_{budget:g}').glob('*/run.json'):
                run=read_json(path);key=(run['model']['name'],run['seed'])
                if key not in expected or key in found:raise RuntimeError('Undeclared/duplicate dense fit in source cell')
                if not internal and not (path.parent/'target_development.json').exists():continue
                found[key]=(path,run)
            if set(found)!=expected:
                pending.append({'fold':fold['name'],'budget':budget,'complete_records':len(found),'required_records':len(expected)})
                continue
            ids=fold['source_validation_ids'] if internal else fold['target_development_ids'];models=[];bindings=[]
            for family in families:
                fits=[]
                for seed in settings['seeds']:
                    path,run=found[(family,seed)];folder=path.parent
                    if run['study_id']!=settings['study_id'] or sha256(folder/'best.pt')!=run['checkpoint_sha256']:
                        raise RuntimeError('Complete dense model identity changed')
                    proof=run['checkpoint_reload']
                    if not proof['fresh_model'] or proof['logit_max_abs_error']>2e-5 or proof['sampled_pixel_probability_max_abs_error']>2e-5:
                        raise RuntimeError('Require verified dense source reloads')
                    filename='validation_predictions.npz' if internal else 'target_development_predictions.npz'
                    with np.load(folder/filename,allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['ids'],ids)
                        metrics=classification(saved['y'],saved['logits'],run['source_logit_threshold'])
                    evaluated=run if internal else read_json(folder/'target_development.json')
                    original=run['validation_metrics'] if internal else evaluated['metrics']
                    if metrics!=original:raise RuntimeError('Dense point metrics differ from saved scores')
                    local=run['validation_localization'] if internal else evaluated['localization']
                    values={'image_AUROC':metrics['auc'],'native_pixel_AP':local['pooled_pixel_average_precision'],
                        'positive_image_Dice':local['mean_positive_image_dice'],'positive_image_IoU':local['mean_positive_image_iou'],
                        'image_recall':metrics['recall'],'review_fraction':metrics['review_rate']}
                    fits.append({'seed':seed,'metrics':values,'training_images':run['training_images'],
                        'validation_images':run['validation_images'],'run':str(path.relative_to(ROOT)),
                        'run_sha256':sha256(path),'checkpoint_sha256':run['checkpoint_sha256'],
                        'predictions':str((folder/filename).relative_to(ROOT)),'predictions_sha256':sha256(folder/filename)})
                    bindings.append({'run':str(path.relative_to(ROOT)),'sha256':sha256(path),
                        'predictions_sha256':sha256(folder/filename),
                        'evaluation_sha256':sha256(folder/'target_development.json') if not internal else sha256(path)})
                    base=found[('dinov3_dense_frozen',seed)][1]
                    if run['training_ids']!=base['training_ids'] or run['validation_ids']!=base['validation_ids']:
                        raise RuntimeError('Matched pixel-supervision identities differ')
                models.append({'family':family,'fits':fits,'mean':{k:float(np.mean([f['metrics'][k] for f in fits])) for k in fits[0]['metrics']},
                    'sample_sd':{k:float(np.std([f['metrics'][k] for f in fits],ddof=1)) for k in fits[0]['metrics']}})
            identity={'protocol_sha256':sha256(SOURCE/'protocol.json'),'fold':fold['name'],'budget':budget,'bindings':bindings}
            cell_id=digest(identity)[:20];destination=FOLDER/cell_id;destination.mkdir(exist_ok=True)
            cell={'cell_id':cell_id,'identity':identity,'fold':fold['name'],'budget':budget,
                'evaluation_population':'source validation reused for checkpoint selection' if internal else 'registered acquisition-held-out development; source-frozen decisions',
                'images':len(ids),'models':models,'scope':'All declared models and3 paired fitted seeds; descriptive development means/SD, not independent confirmation.'}
            cell_path=destination/'cell.json'
            if cell_path.exists():
                if read_json(cell_path)!=cell:raise RuntimeError('Preserve completed dense source cell')
            else:save_json(cell_path,cell)
            complete.append({**cell,'cell_path':str(cell_path.relative_to(ROOT)),'cell_sha256':sha256(cell_path)})
    aggregate={'at':now(),'protocol_sha256':sha256(SOURCE/'protocol.json'),'complete_cells':complete,'pending_cells':pending,
        'all_planned_cells':10,'scope':'Every declared system/seed must complete before a cell appears; no favorable partial-seed summary. Original117/507/PVEL confirmation scores unopened.'}
    save_json(BASE/'dense_development_complete_cells.json',aggregate)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    panels=[]
    for cell in complete:
        path=ROOT/cell['cell_path'];directory=path.parent
        plotted=['image_AUROC','native_pixel_AP','positive_image_Dice']
        lower=min(0.,min(m['mean'][k]-m['sample_sd'][k] for m in cell['models'] for k in plotted))-.025
        upper=max(1.,max(m['mean'][k]+m['sample_sd'][k] for m in cell['models'] for k in plotted))+.025
        fig,axes=plt.subplots(1,3,figsize=(11.7,5.7),sharey=True)
        for ax,key,title in zip(axes,['image_AUROC','native_pixel_AP','positive_image_Dice'],['Image ranking','Native crack pixel AP','Positive-image overlap']):
            for y,model in enumerate(cell['models']):
                family=model['family'];color='#b17a36' if 'reference' in family else '#7955a8' if 'lora' in family else '#26768a' if 'dinov3' in family else '#718892'
                points=[f['metrics'][key] for f in model['fits']]
                ax.scatter(points,np.full(3,y)+[-.08,0,.08],s=14,c=color,alpha=.45)
                ax.errorbar(model['mean'][key],y,xerr=model['sample_sd'][key],fmt='o',c=color,ms=4,capsize=3,lw=1.2)
            ax.set_xlim(lower,upper);ax.set_xticks([0,.2,.4,.6,.8,1.]);ax.set_ylim(len(cell['models'])-.4,-.6)
            ax.set_yticks(range(len(cell['models'])),[NAMES[m['family']] for m in cell['models']])
            ax.set_title(title,loc='left',fontsize=10.5);ax.grid(axis='x',alpha=.15)
            ax.spines[['top','right']].set_visible(False);ax.set_xlabel(key.replace('_',' '))
        fig.suptitle(f"Complete dense cell: {cell['fold']} · {int(cell['budget']*100)}% train masks · {cell['images']} evaluation images",x=.26,ha='left',fontsize=12,color='#193e48',y=.97)
        fig.subplots_adjust(left=.26,right=.98,top=.83,bottom=.24,wspace=.14)
        fig.text(.26,.13,cell['evaluation_population']+'. All required systems and three paired fitted seeds shown.',fontsize=8.2,color='#536f79')
        fig.text(.26,.055,'Mean ± sample SD; seed points retained. Source validation masks and UCF upstream calibration are additional annotation exposure.\n'
            'No matched published SSL result or final confirmation is implied. Pooled pixel AP and image-mean Dice are different endpoints.',fontsize=8,color='#536f79',linespacing=1.5)
        for ext in ['png','pdf','svg']:fig.savefig(directory/f'comparison.{ext}',dpi=230,facecolor='white')
        plt.close(fig)
        save_json(directory/'plot_source.json',{'generator_sha256':sha256(__file__),'cell_sha256':sha256(path),
            'common_x_limits':[lower,upper],'all_symmetric_sample_SD_endpoints_visible':True,
            'render_context':'Matplotlib defaults reset at public entrypoint before applying fixed plot style; independent of caller style.',
            'scope':'Common scale across endpoints; includes0–1 and every full symmetric SD bar without clipping.'})
        panels.append(f'<section><h2>{html.escape(cell["fold"])} · {int(cell["budget"]*100)}% training masks</h2><p>{html.escape(cell["evaluation_population"])}</p><img src="{cell["cell_id"]}/comparison.png" alt="All completed matched dense systems and seeds"><p><a href="{cell["cell_id"]}/comparison.pdf">Vector PDF</a> · <a href="{cell["cell_id"]}/cell.json">Every seed and input hash</a></p></section>')
    (FOLDER/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete real SCDD dense cells</title><style>body{font:16px/1.6 system-ui;color:#193e48;background:#f3f6f7;margin:0}main{max-width:1160px;margin:40px auto;padding:25px;background:white}img{width:100%}section{border-top:1px solid #dbe5e8;margin-top:24px;padding-top:18px}a{color:#216f82}</style><main><h1>Real SCDD matched dense development</h1><p>Only a complete cell with every predeclared model and all three fitted seeds appears below. Internal validation selects checkpoints; acquisition holdouts are development targets. Annotation budgets count training masks separately from source-validation and upstream UCF masks.</p>'''+''.join(panels)+f'<p>{len(complete)} of10 complete source cells. Other cells remain pending; their partial seeds are not displayed as finished comparisons.</p></main></html>')
    print('Complete matched dense source cells',len(complete),'of10; pending',len(pending),flush=True)
    return aggregate


def main():
    # The main dashboard also renders other plots with their own global styles.
    # Reset every default so an unchanged cell has the same PNG in either path.
    with plt.rc_context(matplotlib.rcParamsDefault):
        return _build_complete_cells()


if __name__=='__main__':main()
