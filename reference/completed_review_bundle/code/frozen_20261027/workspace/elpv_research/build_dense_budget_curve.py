"""All four completed internal mask budgets; descriptive source curves only."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now

BASE=ROOT/'artifacts/publication_20261017'
OUT=BASE/'dense_budget_curve_v1'
ARMS=[('deeplabv3plus_full','CNN','#7c8e95'),('dinov3_dense_frozen','DINO frozen','#31869a'),
      ('dinov3_dense_lora','DINO LoRA','#7955a8'),('dinov3_dense_full','DINO full','#287466'),
      ('dinov3_dense_reference','Normal reference','#b98135')]


def main():
    cells=[c for c in read_json(BASE/'dense_development_complete_cells.json')['complete_cells'] if c['fold']=='internal']
    cells=sorted(cells,key=lambda r:r['budget'])
    if [r['budget'] for r in cells]!=[.05,.1,.2,1.]:
        raise RuntimeError('Require every internal mask budget, not a favorable partial curve')
    bindings=[{'path':c['cell_path'],'sha256':c['cell_sha256']} for c in cells]
    for row in bindings:
        if sha256(ROOT/row['path'])!=row['sha256']:raise RuntimeError('Complete dense source cell changed')
    identity={'cell_bindings':bindings,'generator_sha256':sha256(__file__),
        'arms':[a[0] for a in ARMS],'seed_rule':'All3 fitted seeds at every one of four budgets',
        'component_rule':'No-position and mean-evidence are registered only at20/100%; retain separately in all-cell/component reports, not an imputed four-budget curve.'}
    folder=OUT/digest(identity)[:20];folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'source.json'
    if destination.exists():
        if read_json(destination)['identity']!=identity:raise RuntimeError('Preserve existing budget-curve snapshot')
    else:
        x=[]
        for cell in cells:
            counts={f['training_images'] for m in cell['models'] for f in m['fits']}
            if len(counts)!=1 or cell['images']!=120:raise RuntimeError('Common source cohort/mask budgets differ')
            x.append(next(iter(counts)))
            for name,_,_ in ARMS:
                arm=next(m for m in cell['models'] if m['family']==name)
                if sorted(f['seed'] for f in arm['fits'])!=[11,22,33]:raise RuntimeError('Incomplete fitted-seed set')
        style={'font.family':'DejaVu Sans','font.size':8.2,'axes.spines.top':False,
            'axes.spines.right':False,'axes.edgecolor':'#97aab0','axes.labelcolor':'#294953',
            'xtick.color':'#3a555e','ytick.color':'#3a555e','pdf.fonttype':42,'svg.fonttype':'none'}
        endpoints=[('image_AUROC','a  Image ranking','AUROC',(.5,1.)),
            ('native_pixel_AP','b  Native pixel ranking','AP',(0,.35)),
            ('positive_image_Dice','c  Positive-image overlap','Dice',(-.03,.30))]
        with plt.rc_context(matplotlib.rcParamsDefault),plt.rc_context(style):
            fig,axes=plt.subplots(1,3,figsize=(7.2,3.75));handles=[]
            for name,label,color in ARMS:
                models=[next(m for m in c['models'] if m['family']==name) for c in cells]
                for j,(endpoint,title,ylabel,limits) in enumerate(endpoints):
                    means=np.array([m['mean'][endpoint] for m in models]);sd=np.array([m['sample_sd'][endpoint] for m in models])
                    values=[f['metrics'][endpoint] for m in models for f in m['fits']]
                    if min([*(means-sd),*values])<limits[0] or max([*(means+sd),*values])>limits[1]:
                        raise RuntimeError('Do not clip any declared seed/SD endpoint')
                    h=axes[j].errorbar(x,means,yerr=sd,color=color,lw=1.25,marker='o',markersize=3.2,
                        capsize=2,elinewidth=.7,alpha=.9,label=label)
                    for xx,m in zip(x,models):axes[j].scatter([xx]*3,[f['metrics'][endpoint] for f in m['fits']],s=11,color=color,alpha=.3,edgecolors='none')
                    if j==0:handles.append(h)
            for ax,(endpoint,title,ylabel,limits) in zip(axes,endpoints):
                ax.set_xscale('log');ax.set_xlim(14,460);ax.set_xticks(x,[f'{n}\n({100*c["budget"]:g}%)' for n,c in zip(x,cells)])
                ax.tick_params(axis='x',labelsize=7)
                ax.minorticks_off();ax.set_ylim(*limits);ax.set_title(title,loc='left',fontsize=8.6,pad=10)
                ax.set_xlabel('Training masks',labelpad=4);ax.set_ylabel(ylabel,labelpad=3)
                ax.grid(color='#e5ecee',lw=.6);ax.set_axisbelow(True)
                if limits[0]<0:ax.axhline(0,color='#9eafb5',lw=.7)
            fig.suptitle('All internal dense budgets · reused source validation',fontsize=10,y=.98,color='#284e5b')
            fig.subplots_adjust(left=.075,right=.985,top=.84,bottom=.34,wspace=.44)
            fig.legend(handles,[a[1] for a in ARMS],loc='center',bbox_to_anchor=(.53,.17),ncol=5,frameon=False,fontsize=7.2,
                       handlelength=1.5,columnspacing=1.05)
            fig.text(.075,.075,'Three fitted seeds; mean ± sample SD. The same 120 validation images select checkpoints across budgets.',fontsize=7,color='#536f79')
            fig.text(.075,.025,'Training masks are additional to 120 validation masks and 574 upstream UCF calibration annotations.',fontsize=7,color='#536f79')
            for ext in ['png','pdf','svg']:fig.savefig(folder/f'dense_budget_curve.{ext}',dpi=300,facecolor='white')
            plt.close(fig)
        save_json(destination,{'at':now(),'identity':identity,'training_masks':x,'source_validation_images':120,
            'included_fits':60,'all_declared_internal_fits':72,
            'axis_limits':{e[0]:list(e[3]) for e in endpoints},
            'source_cells':[{'budget':c['budget'],'cell_id':c['cell_id'],'models':[{k:m[k] for k in ['family','mean','sample_sd']} for m in c['models']]} for c in cells],
            'figures':[{'path':str((folder/f'dense_budget_curve.{ext}').relative_to(ROOT)),'sha256':sha256(folder/f'dense_budget_curve.{ext}')} for ext in ['png','pdf','svg']],
            'limits':['Descriptive repeated source validation, not independent confirmation or a causal label-cost experiment.',
                'All five fully crossed arms and all fitted seeds; both component arms remain in their complete registered two-budget comparisons.',
                'The small negative Dice plotting margin retains a symmetric SD endpoint below0; observed Dice values are nonnegative.',
                'SD bars describe fitted-seed variability, not confidence intervals. No pixel bootstrap or new fitting/inference.']})
    save_json(OUT/'latest.json',{'at':now(),'source':str(destination.relative_to(ROOT)),'sha256':sha256(destination)})
    print('All four internal dense budgets plotted:5 fully crossed arms×3 seeds;12 component fits retained separately')


if __name__=='__main__':main()
