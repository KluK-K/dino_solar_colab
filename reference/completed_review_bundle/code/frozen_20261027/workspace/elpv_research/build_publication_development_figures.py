"""Plot only complete, paired development cells; missing results stay missing."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT, read_json, save_json, now, sha256


BASE=ROOT/'artifacts/publication_20261017'
LABELS={'dinov3_frozen':'DINOv3-S frozen','dinov3_lora':'DINOv3-S LoRA',
        'dinov3_full':'DINOv3-S full','resnet18_full':'ResNet-18 full',
        'resnet50_full':'ResNet-50 full','efficientnet_b0_full':'EfficientNet-B0 full'}
STYLES=[('#9982bc','o'),('#694192','s'),('#c39acb','^'),('#1d6375','D'),('#4e9bad','v'),('#8cbbbe','P')]


def main():
    study=BASE/'scdd_development_v1';protocol=read_json(study/'protocol.json')
    files=list((study/'runs/internal').glob('budget_*/*/run.json'))
    records=[(read_json(p),p) for p in files];families=list(LABELS)
    budgets=[.05,.1,.2,1.];complete=[];cells=[]
    for budget in budgets:
        group=[(r,p) for r,p in records if r['training_budget_fraction']==budget]
        expected={(family,seed) for family in families for seed in [11,22,33]}
        if {(r['model']['name'],r['seed']) for r,p in group}!=expected:continue
        complete.append(budget)
        for family in families:
            rows=sorted([(r,p) for r,p in group if r['model']['name']==family],key=lambda pair:pair[0]['seed'])
            cells.append({'family':family,'budget':budget,'training_images':rows[0][0]['training_images'],
                'validation_images':rows[0][0]['validation_images'],
                'seeds':[r['seed'] for r,p in rows],
                'AUROC':[r['validation_metrics']['auc'] for r,p in rows],
                'AP':[r['validation_metrics']['average_precision'] for r,p in rows],
                'run_files':[{ 'path':str(p.relative_to(ROOT)), 'sha256':sha256(p)} for r,p in rows]})
    if not cells:return None
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8b9ca3'})
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.25))
    for family,(color,marker) in zip(families,STYLES):
        group=[r for r in cells if r['family']==family]
        for ax,metric in zip(axes,['AUROC','AP']):
            x=[budgets.index(r['budget']) for r in group]
            mean=np.array([np.mean(r[metric]) for r in group]);sd=np.array([np.std(r[metric],ddof=1) for r in group])
            ax.errorbar(x,mean,yerr=sd,color=color,marker=marker,lw=1.6,ms=5,capsize=3,label=LABELS[family])
            for j,offset in enumerate([-.06,0,.06]):
                ax.scatter(np.array(x)+offset,[r[metric][j] for r in group],color=color,s=10,alpha=.28,zorder=2)
    for i,(ax,label) in enumerate(zip(axes,['Image AUROC','Image average precision'])):
        ax.set_ylim(.4,1.01);ax.set_xlim(-.25,3.25)
        ax.set_xticks(range(4),['5%\n18 images','10%\n36 images','20%\n70 images','100%\n351 images'])
        ax.set_xlabel('Training-label availability (120 validation labels additional)',fontsize=9)
        ax.set_ylabel(label);ax.grid(axis='y',color='#e6edef');ax.set_axisbelow(True)
        ax.set_title(chr(97+i)+'  '+label+' on source validation',loc='left',fontsize=11,pad=14)
        for j,budget in enumerate(budgets):
            if budget not in complete:
                ax.axvspan(j-.25,j+.25,color='#f2f5f6',zorder=0)
                ax.text(j,.96,'Pending',ha='center',va='top',color='#8b9ca3',fontsize=8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.89,bottom=.32,wspace=.28)
    fig.text(.08,.035,'Dots: paired seeds 11/22/33; bars: sample SD. Reused source validation selects checkpoints; this is exploratory evidence.',fontsize=8,color='#55717a')
    fig.text(.08,.005,'Only cells complete for every model are plotted. Image labels only; no independent confirmation or physical-module guarantee.',fontsize=8,color='#55717a')
    out=BASE/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'scdd_source_budget_progress.{ext}',dpi=250,facecolor='white')
    plt.close(fig)
    record={'at':now(),'scope':'Exploratory source validation; incomplete cells are omitted',
        'protocol_sha256':sha256(study/'protocol.json'),'complete_budgets':complete,'cells':cells,
        'limits':['Source validation is reused for checkpoint/threshold selection.',
                  'Three training/subset seeds are plotted; SD is not a confidence interval.',
                  'No independent confirmation scores or field prevalence claim.']}
    save_json(out/'scdd_source_budget_progress_source.json',record)
    print('Complete paired budget cells:',complete,'figure:',out/'scdd_source_budget_progress.png')
    return record


if __name__=='__main__':main()
