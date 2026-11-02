"""Does pooled development AUROC hide acquisition-source/label association?"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from elresearch.common import ROOT,read_json,save_json,now,sha256
from elresearch.source_diagnostics import source_summary,source_prevalence_scores


BASE=ROOT/'artifacts/publication_20261017'


def main():
    study=BASE/'scdd_development_v1'
    by_id={r['id']:r for r in read_json(ROOT/'data/scdd_publication_v1/manifest.json')['records'] if r['partition'] in {'train','validation'}}
    snapshot=read_json(BASE/'figures/scdd_source_budget_progress_source.json')
    selection_path=BASE/'calibration_v1/selected_hyperparameters.json'
    selected=read_json(selection_path)['selected']
    cnn=max([r for r in selected if not r['specification']['family'].startswith('dinov3')],
            key=lambda r:np.mean([s['auroc'] for s in r['runs']]))
    cnn_family=cnn['specification']['family']
    cnn_label={'resnet18_full':'ResNet-18 full','resnet50_full':'ResNet-50 full','efficientnet_b0_full':'EfficientNet-B0 full'}[cnn_family]
    budgets=snapshot['complete_budgets'];records=[];priors={}
    for path in sorted((study/'runs/internal').glob('budget_*/*/run.json')):
        run=read_json(path)
        if run['training_budget_fraction'] not in budgets:continue
        validation=[by_id[i] for i in run['validation_ids']]
        with np.load(path.parent/'validation_predictions.npz',allow_pickle=False) as predictions:
            np.testing.assert_array_equal(predictions['ids'],run['validation_ids'])
            y=predictions['y'];score=predictions['logits']
            np.testing.assert_array_equal(y,[r['target'] for r in validation])
        sources=[r['acquisition_source'] for r in validation]
        records.append({'family':run['model']['name'],'seed':run['seed'],'budget':run['training_budget_fraction'],
                        'metrics':source_summary(y,score,sources),'run_path':str(path.relative_to(ROOT)),'run_sha256':sha256(path)})
        key=(run['training_budget_fraction'],run['seed'])
        if key not in priors:
            training=[by_id[i] for i in run['training_ids']]
            prediction,fit=source_prevalence_scores(training,validation)
            priors[key]={'budget':key[0],'seed':key[1],'fit':fit,'metrics':source_summary(y,prediction,sources)}
    result={'at':now(),'scope':'Post-hoc exploratory source-validation diagnostic; no confirmation images scored',
        'complete_budgets':budgets,'models':records,'metadata_only_source_prior':list(priors.values()),
        'shown_CNN_selection':{'family':cnn_family,'UCF_source_hyperparameter_selection_sha256':sha256(selection_path),
                               'criterion':'largest mean UCF source-validation AUROC among the three predeclared CNN families'},
        'analysis_code_sha256':sha256(__file__),'diagnostic_code_sha256':sha256(ROOT/'elresearch/source_diagnostics.py'),
        'limits':['Pooled AUROC and equal-source within-source AUROC answer different population-weighting questions.',
                  'A gap can reflect source/label association and calibration differences; it does not prove a particular model causally uses lab identity.',
                  'Metadata prior is a diagnostic, not an image classifier. It sees only selected training labels.',
                  'Same validation images selected neural checkpoints; these are not blind test outcomes.']}
    save_json(BASE/'source_shortcut_diagnostic.json',result)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.6,5.1))
    counts={.05:18,.1:36,.2:70,1.:351}
    families=[('dinov3_frozen','DINOv3-S frozen','#9982bc','o'),
              ('dinov3_lora','DINOv3-S LoRA','#694192','s'),
              (cnn_family,cnn_label,'#4e9bad','^')]
    for ax,metric,title in zip(axes,['pooled_auroc','macro_within_source_auroc'],
                              ['a  Pooled validation AUROC','b  Mean AUROC within each source']):
        for family,label,color,marker in families:
            means=[];errors=[]
            for budget in budgets:
                values=[r['metrics'][metric] for r in records if r['family']==family and r['budget']==budget]
                means.append(np.mean(values));errors.append(np.std(values,ddof=1))
            ax.errorbar([counts[b] for b in budgets],means,yerr=errors,label=label,color=color,marker=marker,capsize=3,lw=1.7)
        means=[];errors=[]
        for budget in budgets:
            values=[r['metrics'][metric] for r in priors.values() if r['budget']==budget]
            means.append(np.mean(values));errors.append(np.std(values,ddof=1))
        ax.errorbar([counts[b] for b in budgets],means,yerr=errors,label='Source identity only (diagnostic)',color='#777f84',marker='x',ls='--',capsize=3,lw=1.3)
        ax.set_xscale('log');ax.set_xticks([counts[b] for b in budgets],[str(counts[b]) for b in budgets])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(.4,1.01);ax.grid(axis='y',color='#e6edef');ax.set_axisbelow(True)
        ax.set_title(title,loc='left',fontsize=11,pad=14);ax.set_ylabel('AUROC')
        ax.set_xlabel('Training images (log scale; +120 validation labels)',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.3,wspace=.25)
    fig.text(.08,.04,'Means ± sample SD across paired seeds/subsets. CNN shown is preselected by UCF calibration; all six systems are in the JSON.',fontsize=8,color='#55717a')
    fig.text(.08,.015,'Post-hoc development diagnostic; no independent confirmation. A pooled/within-source gap does not identify causal shortcut use.',fontsize=8,color='#55717a')
    for ext in ['png','pdf','svg']:fig.savefig(BASE/'figures'/f'source_shortcut_diagnostic.{ext}',dpi=250,facecolor='white')
    plt.close(fig)
    for budget in budgets:
        print('Budget',budget)
        for family in ['dinov3_frozen','dinov3_lora',cnn_family]:
            row=[r for r in records if r['budget']==budget and r['family']==family]
            print(family,'pooled',round(np.mean([r['metrics']['pooled_auroc'] for r in row]),4),
                  'within-source',round(np.mean([r['metrics']['macro_within_source_auroc'] for r in row]),4))
    return result


if __name__=='__main__':main()
