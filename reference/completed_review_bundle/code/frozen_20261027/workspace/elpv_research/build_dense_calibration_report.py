"""Complete source-calibration figures from saved native maps; no inference.

Selection is the registered three-seed UCF rule. Qualitative maps use fixed
seed11 and annotation/identity-selected examples, never ranked by model success.
"""
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch.data import records,load_image,load_mask,IGNORE

BASE=ROOT/'artifacts/publication_20261017'
SOURCE=BASE/'dense_development_v1'
OUTPUT=BASE/'dense_source_calibration_analysis_v1'
NAMES={'deeplabv3plus_full':'DeepLabv3+ / ResNet-50',
       'dinov3_dense_frozen':'DINOv3 frozen decoder',
       'dinov3_dense_lora':'DINOv3 + LoRA decoder',
       'dinov3_dense_full':'DINOv3 full adaptation'}


def outline(ax,mask,score=None):
    truth=(mask==1)
    if truth.any() and not truth.all():
        ax.contour(truth,levels=[.5],colors=['#36dbd0'],linewidths=.7)
    if score is not None:
        pred=(score>.5)&(mask!=IGNORE)
        if pred.any() and not pred.all():
            ax.contour(pred,levels=[.5],colors=['white'],linewidths=.6,linestyles='--')


def roi(mask):
    h,w=mask.shape;yy,xx=np.where(mask==1)
    if len(xx):
        x0,x1,y0,y1=xx.min(),xx.max()+1,yy.min(),yy.max()+1
        cx,cy=(x0+x1)/2,(y0+y1)/2;side=max(96,max(x1-x0,y1-y0)*1.3)
    else:
        cx,cy=w/2,h/2;side=max(96,min(h,w)*.45)
    width,height=min(w,int(np.ceil(side))),min(h,int(np.ceil(side)))
    left=max(0,min(w-width,int(round(cx-width/2))));top=max(0,min(h-height,int(round(cy-height/2))))
    return left,top,left+width,top+height


def main():
    selection_path=SOURCE/'selected_hyperparameters.json'
    if not selection_path.exists():
        print('Dense source calibration pending; no partial comparison exported',flush=True);return
    settings=read_json(SOURCE/'protocol.json');selection=read_json(selection_path)
    trials=list((SOURCE/'calibration').glob('*/run.json'))
    if len(trials)!=24:raise RuntimeError('Require all24 source-calibration fits')
    by={r['id']:r for r in records('ucf','validation')}
    if set(by)!=set(settings['calibration']['validation_ids']):raise RuntimeError('Source validation identities changed')
    models=[];bindings=[];grid=[]
    for candidate in settings['calibration']['candidates']:
        values=[]
        for seed in [11,22,33]:
            path=SOURCE/'calibration'/f"{candidate['name']}_seed{seed}"/'run.json'
            run=read_json(path)
            values.append(run['validation_localization']['pooled_pixel_average_precision'])
        grid.append({'candidate':candidate,'native_pixel_AP':values,
            'mean_native_pixel_AP':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))})
    for chosen in selection['selected']:
        candidate=next(c for c in settings['calibration']['candidates'] if c['family']==chosen['family'] and c['encoder_lr']==chosen['encoder_lr'] and c['head_lr']==chosen['head_lr'])
        winner=max((g for g in grid if g['candidate']['family']==chosen['family']),key=lambda g:g['mean_native_pixel_AP'])
        if winner['candidate']!=candidate:raise RuntimeError('Selected source rate differs from registered mean-AP rule')
        seed_runs=[]
        for seed in [11,22,33]:
            folder=SOURCE/'calibration'/f"{candidate['name']}_seed{seed}"
            run=read_json(folder/'run.json')
            if run['study_id']!=settings['study_id'] or sha256(folder/'best.pt')!=run['checkpoint_sha256']:
                raise RuntimeError('Source checkpoint identity mismatch')
            if not run['checkpoint_reload']['fresh_model']:
                raise RuntimeError('Require a fresh source model reload')
            seed_runs.append(run)
            bindings.append({'run':str((folder/'run.json').relative_to(ROOT)),'sha256':sha256(folder/'run.json'),
                             'checkpoint_sha256':run['checkpoint_sha256']})
        folder=SOURCE/'calibration'/f"{candidate['name']}_seed11"
        ids=seed_runs[0]['validation_localization']['fixed_example_ids'];maps={}
        for identifier in ids:
            path=folder/'fixed_validation_examples'/(digest(identifier)+'.npz')
            with np.load(path,allow_pickle=False) as saved:
                if str(saved['id'])!=identifier:raise RuntimeError('Native-map identity mismatch')
                maps[identifier]=saved['score'].astype(np.float32)
            bindings.append({'map':str(path.relative_to(ROOT)),'sha256':sha256(path)})
        models.append({'family':chosen['family'],'selected_specification':chosen,'runs':seed_runs,'maps':maps,
            'summary':{'image_AUROC':[r['validation_metrics']['auc'] for r in seed_runs],
                'native_pixel_AP':[r['validation_localization']['pooled_pixel_average_precision'] for r in seed_runs],
                'positive_image_Dice':[r['validation_localization']['mean_positive_image_dice'] for r in seed_runs]}})
    if len(models)!=4 or any(set(m['maps'])!=set(models[0]['maps']) for m in models):
        raise RuntimeError('Require identical fixed native examples across all four architectures')
    ids=set(models[0]['maps']);negative=sorted(i for i in ids if not by[i]['target'])
    positive=sorted((i for i in ids if by[i]['target']),key=lambda i:(by[i]['crack_area_fraction'],i))
    ordered=negative+positive;main_ids=[negative[0],positive[0],positive[-1]]
    OUTPUT.mkdir(exist_ok=True)
    plan={'version':2,'kind':'Complete selected source-calibration visualization; no new inference',
        'protocol_sha256':sha256(SOURCE/'protocol.json'),'source_selection_sha256':sha256(selection_path),
        'generator_sha256':sha256(__file__),'source_bindings':bindings,'seed_for_qualitative_maps':11,
        'amendment_sha256':sha256(OUTPUT/'amendment.json'),
        'main_ids':main_ids,'all_fixed_ids':ordered,
        'selection':'All8 original annotation/identity-based fixed cases. Main: lexicographically first negative and smallest/largest annotated-area positive. Never chosen by model score.',
        'roi':'Square annotation bounding box plus30% context, minimum96 native pixels, clipped to image; centered45%-shorter-edge view for negative. Same ROI for every model.',
        'display':'Raw EL0–255 and softmax0–1 fixed scales, cyan GT outline, white dashed score>0.5 contour, ignored annotation gray. Stored visualization maps float16; research metrics taken from original full-float native evaluation.',
        'limits':'Selected reused source-validation data, not independent test; seed11 is illustrative, not an ensemble or representative success sample. The supervised adapted DeepLabv3+ is not the full PV-S3 semi-supervised method.'}
    if (OUTPUT/'plan.json').exists() and read_json(OUTPUT/'plan.json')!=plan:
        raise RuntimeError('Archive the previous figure specification before changing it')
    if not (OUTPUT/'plan.json').exists():save_json(OUTPUT/'plan.json',plan)
    native={i:(np.asarray(load_image(by[i]).convert('L')),load_mask(by[i])) for i in ordered}
    for identifier,(image,mask) in native.items():
        if image.shape!=mask.shape or any(m['maps'][identifier].shape!=mask.shape for m in models):
            raise RuntimeError('Native image, truth and map geometry differ')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#ccd4d8')

    def gallery(identifiers,zoom,name):
        height=2.1*len(identifiers)+1.6
        fig,axes=plt.subplots(len(identifiers),6,figsize=(12.4,height),squeeze=False)
        for row,identifier in enumerate(identifiers):
            image,mask=native[identifier];left,top,right,bottom=roi(mask)
            crop=(slice(top,bottom),slice(left,right))
            axes[row,0].imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            if zoom:
                axes[row,0].add_patch(plt.Rectangle((left,top),right-left,bottom-top,fill=False,edgecolor='white',lw=.9))
                axes[row,1].imshow(image[crop],cmap='gray',vmin=0,vmax=255,interpolation='nearest');outline(axes[row,1],mask[crop])
            else:
                colors=ListedColormap(['#20292f','#36dbd0','#e9993c','#ccd4d8'])
                labels=np.where(mask==IGNORE,3,mask)
                axes[row,1].imshow(labels,cmap=colors,norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),interpolation='nearest')
            axes[row,0].set_ylabel(identifier.split('/')[-1],fontsize=7.5)
            for column,model in enumerate(models,2):
                score=model['maps'][identifier];display_mask=mask
                if zoom:score=score[crop];display_mask=mask[crop]
                axes[row,column].imshow(np.ma.masked_where(display_mask==IGNORE,score),cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
                outline(axes[row,column],display_mask,score)
            for ax in axes[row]:
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
        headings=['Native EL','EL + crack truth (ROI)' if zoom else 'Canonical annotation']+[NAMES[m['family']].replace(' / ','\n').replace(' decoder','\ndecoder').replace(' adaptation','\nadaptation') for m in models]
        for ax,title in zip(axes[0],headings):ax.set_title(title,fontsize=8.5,pad=9)
        fig.subplots_adjust(left=.09,right=.98,top=1-.9/height,bottom=1.35/height,wspace=.08,hspace=.14)
        fig.suptitle('Fixed source-validation examples · selected checkpoints · seed11',x=.09,ha='left',fontsize=13,y=.985,color='#193e48')
        cax=fig.add_axes([.55,.75/height,.40,.12/height]);fig.colorbar(ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,.25,.5,.75,1]).set_label('Crack softmax score · fixed [0,1] scale',fontsize=8)
        fig.text(.09,.78/height,'Cyan: crack truth\nWhite dashed: score >0.5',fontsize=8,color='#506d77')
        fig.text(.09,.12/height,'Reused UCF source validation; illustrative saved float16 maps, not confirmation. Gray: ignored annotation. '+('Identical GT/context-defined ROI per model.' if zoom else 'Truth: cyan crack, orange other defect, dark normal.'),fontsize=7.5,color='#506d77')
        for extension in ['png','pdf','svg']:
            fig.savefig(OUTPUT/f'{name}.{extension}',dpi=220,facecolor='white')
        plt.close(fig)
    gallery(main_ids,True,'selected_source_native_crops')
    gallery(ordered,False,'all_fixed_source_native_examples')
    summary=[{'family':m['family'],'selected_specification':m['selected_specification'],
        'all_seed_values':m['summary'],'mean':{k:float(np.mean(v)) for k,v in m['summary'].items()},
        'sample_sd':{k:float(np.std(v,ddof=1)) for k,v in m['summary'].items()}} for m in models]
    save_json(OUTPUT/'summary.json',{'at':now(),'plan_sha256':sha256(OUTPUT/'plan.json'),
        'models':summary,'all8_source_grid_candidates':grid,'fixed_seed11_examples':ordered,'confirmation_scores_opened':False,
        'scope':'Three source calibration seeds per selected system; reused for source rate/checkpoint selection. SD is descriptive, not independent test uncertainty.'})
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in [NAMES[m['family']]]+
        [f"{m['mean'][key]:.3f} ± {m['sample_sd'][key]:.3f}" for key in ['image_AUROC','native_pixel_AP','positive_image_Dice']])+'</tr>' for m in summary)
    (OUTPUT/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete dense source calibration</title><style>body{font:16px/1.6 system-ui;color:#193e48;background:#f3f6f7;margin:0}main{max-width:1180px;margin:40px auto;padding:28px;background:white}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce5e7}a{color:#246e81}.small{font-size:13px;color:#57717c}</style><main>
<h1>Complete24-fit dense source calibration</h1><p>Each architecture has two predeclared learning-rate candidates and three seeds. The registered rule selects the highest mean full-native UCF validation pixel AP. These same144 validation images select rates/checkpoints, so the summary is development evidence.</p>
<table><tr><th>Selected system</th><th>Image AUROC</th><th>Native crack pixel AP</th><th>Positive-image Dice</th></tr>'''+table+'''</table>
<p class="small">Means ± sample SD over the three fitted seeds. Same430 training image/mask identities plus144 validation image/mask identities. Source provenance remains conditional; no electrical failure ground truth. The supervised adapted DeepLabv3+ here does not reproduce full semi-supervised PV-S3.</p>
<h2>Native inputs and shared annotation-defined views</h2><img src="selected_source_native_crops.png" alt="Fixed native EL and identical ground-truth-defined crops with all four selected source models">
<p>The negative case is the first fixed negative ID; positive cases have the smallest/largest annotation area among the existing fixed gallery. Identical crop coordinates apply to every model. No model outcome selects a case or crop. Color scales are fixed, without per-image normalization. Qualitative maps use seed11 and stored float16 scores; quantitative tables use the original full-float native evaluation.</p>
<details><summary>All8 originally fixed validation cases, with no omitted failure</summary><img src="all_fixed_source_native_examples.png" alt="All eight fixed native source examples and every selected architecture"></details>
<p class="small"><a href="selected_source_native_crops.pdf">Main vector PDF</a> · <a href="all_fixed_source_native_examples.pdf">Full gallery PDF</a> · <a href="summary.json">Every seed value</a> · <a href="plan.json">Selection rules and source/map hashes</a></p>
<p>The remaining real SCDD matched localization, normal-reference ablations, semi-supervised method controls and independent confirmation remain separate. Source calibration alone cannot establish paper-level superiority.</p></main></html>''')
    print('Exported complete24-fit dense source summary and fixed native-map figures',flush=True)


if __name__=='__main__':main()
