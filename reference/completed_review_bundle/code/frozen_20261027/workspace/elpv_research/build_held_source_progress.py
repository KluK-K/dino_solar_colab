"""Completed three-seed acquisition holdouts; never include the sealed117 tests."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,now


BASE=ROOT/'artifacts/publication_20261017'
FAMILIES=['dinov3_frozen','dinov3_lora','dinov3_full','resnet18_full','resnet50_full','efficientnet_b0_full']
LABELS=['DINOv3-S frozen','DINOv3-S LoRA','DINOv3-S full','ResNet-18 full','ResNet-50 full','EfficientNet-B0 full']


def main():
    root=BASE/'scdd_development_v1';protocol=read_json(root/'protocol.json');cells=[]
    for fold in protocol['split_manifest']['folds'][1:]:
        for budget in fold['budgets']:
            paths=list((root/'runs'/fold['name']/f'budget_{budget:g}').glob('*/target_development.json'))
            expected={(family,seed) for family in FAMILIES for seed in [11,22,33]};found={}
            for path in paths:
                run=read_json(path.parent/'run.json');found[(run['model']['name'],run['seed'])]=(run,path)
            if set(found)!=expected:continue
            rows=[]
            for family in FAMILIES:
                runs=[]
                for seed in [11,22,33]:
                    run,path=found[(family,seed)];target=read_json(path)
                    with np.load(path.parent/'target_development_predictions.npz',allow_pickle=False) as predictions:
                        np.testing.assert_array_equal(predictions['ids'],fold['target_development_ids'])
                    runs.append({'seed':seed,'metrics':target['metrics'],'training_images':run['training_images'],
                                 'validation_images':run['validation_images'],'run':str((path.parent/'run.json').relative_to(ROOT)),
                                 'run_sha256':sha256(path.parent/'run.json'),'target_metrics_sha256':sha256(path),
                                 'prediction_sha256':sha256(path.parent/'target_development_predictions.npz')})
                rows.append({'family':family,'seeds':runs,
                    'mean':{k:float(np.mean([r['metrics'][k] for r in runs])) for k in ['auc','average_precision','recall','review_rate']},
                    'sample_sd':{k:float(np.std([r['metrics'][k] for r in runs],ddof=1)) for k in ['auc','average_precision','recall','review_rate']}})
            cells.append({'fold':fold['name'],'target_acquisition_source':fold['held_out_acquisition_source'],
                          'training_fraction':budget,'target_images':len(fold['target_development_ids']),
                          'target_positive':rows[0]['seeds'][0]['metrics']['positive'],'models':rows})
    result={'at':now(),'protocol_sha256':sha256(root/'protocol.json'),'complete_cells':cells,
        'scope':'Leave-one-acquisition-source-out DEVELOPMENT among471 images; original117 test images remain sealed',
        'limits':['All three paired seeds and all six systems must be complete for a cell to appear.',
                  'Target recall/review use source-validation thresholds; no target threshold fit.',
                  'Other folds have observed these development sources; this is not final independent confirmation.',
                  'Sample SD describes fitted-seed/subset variability, not a confidence interval or verified physical-module independence.']}
    save_json(BASE/'held_source_development_complete_cells.json',result)
    if not cells:return result
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    first=cells[0];fig,axes=plt.subplots(1,2,figsize=(11,5.6),sharey=True,gridspec_kw={'width_ratios':[1,1.1]})
    for i,row in enumerate(first['models']):
        color='#7753a5' if row['family'].startswith('dinov3') else '#28778a'
        values=[r['metrics']['auc'] for r in row['seeds']]
        axes[0].scatter(values,np.full(3,i)+np.array([-.1,0,.1]),color=color,s=16,alpha=.4)
        axes[0].errorbar(row['mean']['auc'],i,xerr=row['sample_sd']['auc'],color=color,fmt='o',ms=5,capsize=4,lw=1.6)
        for metric,offset,marker,c in [('recall',-.14,'o','#bd6739'),('review_rate',.14,'s','#617985')]:
            vals=[r['metrics'][metric] for r in row['seeds']]
            axes[1].scatter(vals,np.full(3,i)+offset+np.array([-.04,0,.04]),color=c,s=15,alpha=.25,
                            facecolors='none' if metric=='review_rate' else c)
            axes[1].scatter(row['mean'][metric],i+offset,marker=marker,color=c,s=44,
                            facecolors='none' if metric=='review_rate' else c,
                            label=('Target recall' if metric=='recall' else 'Target review fraction') if i==0 else None)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#e7edef');ax.set_axisbelow(True)
    axes[0].set_yticks(range(6),LABELS);axes[0].set_ylim(5.6,-.6);axes[0].set_xlim(.15,1.01)
    axes[0].axvline(.5,color='#aab9bf',lw=1,ls='--');axes[0].set_xlabel('Image AUROC on held source')
    axes[0].set_title('a  Ranking performance',loc='left',fontsize=11,pad=16)
    axes[1].set_xlim(-.02,1.02);axes[1].set_xlabel('Fraction of target images / target positives')
    axes[1].set_title('b  Same source-frozen operating points',loc='left',fontsize=11,pad=16)
    axes[1].axvline(.95,color='#aab9bf',lw=1,ls='--')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.33),frameon=False,ncol=1,fontsize=9)
    fig.suptitle(f"First complete acquisition holdout: {first['target_acquisition_source']} · {first['target_images']} images ({first['target_positive']} crack positives)",
                 fontsize=12,y=.97)
    fig.subplots_adjust(left=.2,right=.98,top=.82,bottom=.27,wspace=.3)
    fig.text(.05,.095,'20% source-training labels; source validation labels additional. Left: three seed points and mean ± sample SD; right: individual points and means.',fontsize=8,color='#55717a')
    fig.text(.05,.055,'The dashed 0.95 line on the right is the SOURCE recall criterion. It is not a guaranteed target recall. No target threshold was fitted.',fontsize=8,color='#55717a')
    fig.text(.05,.015,'Exploratory held-source development, not the sealed 117 test. Capture families are not verified physical module IDs. All completed cells are in the JSON.',fontsize=8,color='#55717a')
    for ext in ['png','pdf','svg']:fig.savefig(BASE/'figures'/f'first_held_source_development.{ext}',dpi=250,facecolor='white')
    plt.close(fig);print('Completed held-source cells:',len(cells),'first:',first['fold'],first['training_fraction'])
    return result


if __name__=='__main__':main()
