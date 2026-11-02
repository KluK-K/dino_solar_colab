"""All prespecified exploratory LoRA contrasts, with conditional family uncertainty."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,now

BASE=ROOT/'artifacts/publication_20261017'
BASELINES=['dinov3_frozen','resnet18_full','resnet50_full','efficientnet_b0_full']
LABELS=['Frozen DINOv3-S','ResNet-18','ResNet-50','EfficientNet-B0']


def main():
    source=BASE/'held_source_paired_uncertainty.json';data=read_json(source)
    records=data['comparisons'];keys=list(dict.fromkeys((r['fold'],r['training_fraction']) for r in records))
    if any(sum(r['fold']==fold and r['training_fraction']==budget for r in records)!=4 for fold,budget in keys):
        raise RuntimeError('Only complete predeclared four-contrast cells may be plotted')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#83969d',
        'axes.labelcolor':'#29434e','xtick.color':'#385461','ytick.color':'#29434e'})
    fig,axes=plt.subplots(1,4,figsize=(12.5,max(3.8,len(keys)*.42+2.1)),sharey=True)
    rows=[];positions=np.arange(len(keys))
    for column,(baseline,label) in enumerate(zip(BASELINES,LABELS)):
        ax=axes[column];ax.axvline(0,color='#718992',ls=(0,(3,3)),lw=1,zorder=1)
        for position,(fold,budget) in enumerate(keys):
            record=next(r for r in records if r['fold']==fold and r['training_fraction']==budget and r['first']==baseline)
            if record['statistics_plan_sha256']!=sha256(BASE/'statistics_plan_v2.json'):
                raise RuntimeError('Paired interval differs from registered statistical implementation')
            result=record['bootstrap']['metrics']['auroc'];mean=result['mean_seed_difference'];lo,hi=result['acquisition_family_percentile_ci95']
            # Plot interval endpoints explicitly; percentile intervals need not bracket the point estimate.
            ax.plot([lo,hi],[position,position],color='#7c58a6',lw=2,zorder=2)
            ax.plot([lo,lo],[position-.07,position+.07],color='#7c58a6',lw=1)
            ax.plot([hi,hi],[position-.07,position+.07],color='#7c58a6',lw=1)
            ax.scatter([mean],[position],s=36,c='#7c58a6',edgecolors='white',lw=.6,zorder=3)
            rows.append({'fold':fold,'training_fraction':budget,'baseline':baseline,
                'difference':mean,'conditional_acquisition_family_ci95':[lo,hi],
                'paired_seed_differences':result['paired_seed_differences']})
        ax.set_xlim(-.10,.55);ax.set_xticks([0,.2,.4]);ax.grid(axis='x',color='#e7edef',lw=.7)
        ax.set_title(f'{chr(97+column)}  LoRA − {label}',loc='left',fontsize=10.5,pad=14)
        ax.tick_params(axis='y',length=0);ax.set_xlabel('AUROC difference',labelpad=9)
        ax.set_ylim(len(keys)-.5,-.7)
    axes[0].set_yticks(positions,[f"{fold.removeprefix('holdout_')}  ·  {int(budget*100)}%" for fold,budget in keys])
    fig.suptitle('Acquisition source and comparator both matter',x=.03,ha='left',fontsize=15,color='#183e4b',y=.97)
    fig.subplots_adjust(left=.13,right=.985,top=.77,bottom=.31,wspace=.20)
    fig.text(.03,.12,'Dots: mean of three paired seed differences. Lines: 95% acquisition-family percentile intervals, 1,000 resamples.',fontsize=9,color='#4a6874')
    fig.text(.03,.055,'Exploratory development, conditional on fitted models. Capture families are not verified modules; these are not adjusted confirmatory tests.',fontsize=8.4,color='#607b85')
    output=BASE/'figures';output.mkdir(exist_ok=True)
    for extension in ['png','pdf','svg']:fig.savefig(output/f'held_source_paired_contrasts.{extension}',dpi=250,facecolor='white')
    plt.close(fig)
    save_json(output/'held_source_paired_contrasts_source.json',{'at':now(),'input_sha256':sha256(source),
        'plot_code_sha256':sha256(__file__),'scope':data['scope'],'contrasts':rows})
    print('All conditional contrasts plotted',len(rows),flush=True)

if __name__=='__main__':main()
