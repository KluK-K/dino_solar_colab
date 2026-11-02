"""All complete three-seed acquisition holdouts; pending cells remain explicit."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,now
from build_held_source_progress import FAMILIES,LABELS


BASE=ROOT/'artifacts/publication_20261017'


def main():
    source=BASE/'held_source_development_complete_cells.json';data=read_json(source)
    sources=['ARTS','CFVS','CSIR'];budgets=[.2,1.];metrics=['auc','recall']
    lookup={(c['target_acquisition_source'],c['training_fraction']):c for c in data['complete_cells']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.6),sharey=True)
    cmap=plt.get_cmap('cividis').copy();cmap.set_bad('#edf1f3')
    for row,metric in enumerate(metrics):
        for col,budget in enumerate(budgets):
            ax=axes[row,col];values=np.full((6,3),np.nan);errors=values.copy()
            for j,origin in enumerate(sources):
                cell=lookup.get((origin,budget))
                if cell is None:continue
                by_family={m['family']:m for m in cell['models']}
                for i,family in enumerate(FAMILIES):
                    values[i,j]=by_family[family]['mean'][metric];errors[i,j]=by_family[family]['sample_sd'][metric]
            image=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,cmap=cmap,aspect='auto')
            for i in range(6):
                for j in range(3):
                    if np.isnan(values[i,j]):
                        ax.text(j,i,'Pending',ha='center',va='center',fontsize=8,color='#81919a')
                    else:
                        color='white' if values[i,j]<.56 else '#173842'
                        ax.text(j,i,f'{values[i,j]:.3f}\n±{errors[i,j]:.3f}',ha='center',va='center',fontsize=9,color=color,linespacing=1.3)
            ax.set_xticks(range(3),sources);ax.set_yticks(range(6),LABELS)
            ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True)
            ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='both',length=0)
            for spine in ax.spines.values():spine.set_visible(False)
            name='Image AUROC' if metric=='auc' else 'Target recall'
            ax.set_title(f"{'abcd'[row*2+col]}  {name} · {int(budget*100)}% labels",loc='left',fontsize=10.5,pad=11)
            if row==1:ax.set_xlabel('Held-out acquisition source',labelpad=9)
    fig.subplots_adjust(left=.24,right=.9,top=.91,bottom=.21,hspace=.4,wspace=.14)
    bar=fig.add_axes([.925,.235,.015,.65]);fig.colorbar(image,cax=bar,label='Mean (fixed 0–1 scale)')
    fig.text(.035,.105,'Each populated cell includes all six models and all three fitted seeds. Values: mean ± sample SD; SD is not a confidence interval.',fontsize=8,color='#56717b')
    fig.text(.035,.065,'Recall uses the source-validation ≥95% recall rule; target labels never select thresholds. Validation labels are additional to each training budget.',fontsize=8,color='#56717b')
    fig.text(.035,.025,'Exploratory development only. Acquisition families are not verified physical modules. Original SCDD117 and external confirmation remain sealed.',fontsize=8,color='#56717b')
    for ext in ['png','pdf','svg']:fig.savefig(BASE/'figures'/f'held_source_development_matrix.{ext}',dpi=250,facecolor='white')
    plt.close(fig);save_json(BASE/'figures/held_source_development_matrix_source.json',{
        'at':now(),'input_sha256':sha256(source),'complete_cells':data['complete_cells'],'scope':data['scope'],
        'scale':'Fixed0..1 for AUROC and recall; pending cells are missing experiments, never zeros'})
    print('All completed holdout cells plotted:',len(data['complete_cells']))


if __name__=='__main__':main()
