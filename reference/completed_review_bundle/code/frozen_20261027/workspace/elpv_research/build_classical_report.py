"""Show every completed classical control alongside the registered LoRA arm."""
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT, read_json, save_json, sha256, now

BASE=ROOT/'artifacts/publication_20261017'
FOLDER=BASE/'classical_development_analysis_v1'
NAMES={'hog_stats_logistic':'HOG + Logistic', 'hog_stats_linear_svm':'HOG + Linear SVM',
       'hog_stats_rbf_svm':'HOG + RBF SVM', 'dinov3_cls_logistic':'DINOv3 CLS + Logistic',
       'dinov3_cls_linear_svm':'DINOv3 CLS + Linear SVM', 'dinov3_cls_rbf_svm':'DINOv3 CLS + RBF SVM',
       'dinov3_lora':'DINOv3 + LoRA'}


def main():
    result=read_json(FOLDER/'results.json');neural=read_json(BASE/'held_source_development_complete_cells.json')
    if result['status']!='complete' or result['verified_selected_model_count']!=186:
        raise RuntimeError('Only show a fully verified complete classical analysis')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(12.6,7.9),sharex=True,sharey=True)
    rows=[]
    for row,budget in enumerate([.2,1.]):
        for col,source in enumerate(['ARTS','CFVS','CSIR']):
            ax=axes[row,col];fold='holdout_'+source
            cell=next(c for c in result['cells'] if c['fold']==fold and c['training_fraction']==budget)
            lora_cell=next(c for c in neural['complete_cells'] if c['fold']==fold and c['training_fraction']==budget)
            models=cell['models']+[next(m for m in lora_cell['models'] if m['family']=='dinov3_lora')]
            for y,model in enumerate(models):
                family=model['family'];mean=model['mean']['auc'];sd=model['sample_sd']['auc']
                color='#a4b2b6' if family.startswith('hog') else '#27778a' if family.startswith('dinov3_cls') else '#7955a8'
                individual=[s['metrics']['auc'] for s in model['seeds']]
                ax.scatter(individual,np.array([y-.08,y,y+.08]),s=11,c=color,alpha=.4,zorder=3)
                ax.errorbar(mean,y,xerr=sd,fmt='o',ms=4.5,c=color,capsize=3,lw=1.1,zorder=4)
                rows.append({'source':source,'budget':budget,'family':family,'mean_auc':mean,'sample_sd':sd,'seed_auc':individual})
            ax.axvline(.5,color='#bac7ca',ls='--',lw=.9,zorder=0)
            ax.set_title(f'{source} · {int(budget*100)}% labels',loc='left',fontsize=10.5,pad=10)
            ax.set_xlim(0,1.02);ax.set_ylim(6.6,-.6);ax.set_xticks([0,.25,.5,.75,1.])
            ax.set_yticks(range(7),list(NAMES.values()));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            if row==1:ax.set_xlabel('Held-source image AUROC')
    fig.suptitle('Frozen embeddings, classical classifiers and encoder adaptation',
                 x=.22,ha='left',fontsize=14,color='#183e49',y=.98)
    fig.subplots_adjust(left=.22,right=.98,top=.89,bottom=.20,wspace=.12,hspace=.35)
    fig.text(.22,.11,'All 6 classical controls and LoRA shown in every source/budget cell. Points: fitted seed/subset records; bars: sample SD.',
             fontsize=9,color='#4d6974')
    fig.text(.22,.068,'At 100% labels, classical predictions are deterministic duplicates, not 3 independent optimizer repetitions.\n'
             'Same source annotation identities; different objectives and tuning budgets. Exploratory development, not independent confirmation.',
             fontsize=8.5,color='#4d6974',linespacing=1.6)
    for extension in ['png','pdf','svg']:
        fig.savefig(BASE/'figures'/f'classical_and_lora_source_holdouts.{extension}',dpi=230,facecolor='white')
    plt.close(fig)
    save_json(BASE/'figures/classical_and_lora_source_holdouts_source.json',{'at':now(),
        'classical_analysis_sha256':sha256(FOLDER/'results.json'),
        'neural_complete_cells_sha256':sha256(BASE/'held_source_development_complete_cells.json'),
        'generator_sha256':sha256(__file__),'all_42_family_cell_summaries':rows,
        'scope':'Same source-training/validation identities verified; no confirmation or new fit.'})
    table=[]
    for cell in result['cells']:
        for model in cell['models']:
            value=model['mean'];table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [
                cell['fold'],f"{int(cell['training_fraction']*100)}%",NAMES[model['family']],
                f"{value['auc']:.3f}",f"{value['recall']:.3f}",f"{value['review_rate']:.3f}"] )+'</tr>')
    contrast=[]
    for row in result['comparisons']:
        value=row['bootstrap']['metrics']['auroc'];low,high=value['acquisition_family_percentile_ci95']
        contrast.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [row['fold'],
            f"{int(row['training_fraction']*100)}%",NAMES[row['first']],f"{value['mean_seed_difference']:+.3f}",
            f"[{low:+.3f}, {high:+.3f}]"] )+'</tr>')
    (FOLDER/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete classical development controls</title>
<style>body{margin:0;background:#f3f6f7;color:#193e48;font:16px/1.6 system-ui}main{max-width:1160px;margin:40px auto;background:white;padding:28px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #dce4e7;padding:8px}a{color:#226d81}.small{font-size:13px;color:#57717b}.scroll{overflow:auto}</style><main>
<h1>All classical controls completed and reloaded</h1>
<p>All70 source-grid trials converged. The6 selected UCF pipelines and180 SCDD development pipelines were reloaded: all186 reproduce saved source logits exactly; the108 held-source predictions also match exactly. Training and validation identities match the corresponding neural experiments.</p>
<img src="../figures/classical_and_lora_source_holdouts.png" alt="All six classical classifiers and LoRA across every acquisition-held-out source and both training budgets">
<p>The frozen DINOv3 CLS + Linear SVM baseline is stronger than the corresponding HOG + Linear SVM in every completed source-held-out mean. LoRA has higher mean AUROC than the DINOv3 CLS + Linear SVM in these cells, but not every pairwise interval excludes zero. This remains retrospective development evidence. Rankings, source-threshold recall and review burden are separate readouts.</p>
<p class="small">At full training budget, repeated classical seed records use identical data and deterministic optimizers; their zero SD is not proof of training robustness. Classical parameter grids and neural learning-rate grids have different sizes. Different pretrained corpora and objectives prevent a causal architecture claim.</p>
<h2>Every source-held-out paired contrast</h2><p>Difference is LoRA minus the named comparator.1,000 paired capture-family bootstrap replicates condition on the fitted models; provided families are not verified physical modules. These36 exploratory contrasts are not multiplicity-adjusted significance tests. No target threshold is fitted.</p>
<div class="scroll"><table><tr><th>Held source</th><th>Training labels</th><th>Comparator</th><th>AUROC difference</th><th>Conditional95% interval</th></tr>'''+''.join(contrast)+'''</table></div>
<h2>All60 classical cell summaries</h2><p class="small">Internal rows use source-validation scores also used for checkpoint/operating-point decisions; held-source rows use the registered target-development split. These populations must not be treated as interchangeable test results.</p>
<div class="scroll"><table><tr><th>Fold</th><th>Training labels</th><th>Model</th><th>AUROC</th><th>Recall</th><th>Review</th></tr>'''+''.join(table)+'''</table></div>
<p><a href="results.json">All individual seed values, source selections, reload hashes and intervals</a> · <a href="all_classical_cells.csv">All cell summaries CSV</a> · <a href="../figures/classical_and_lora_source_holdouts.pdf">Vector figure PDF</a> · <a href="plan.json">Retrospective analysis plan</a></p>
<p class="small">No encoder was run, no model refitted and no final-confirmation model score opened by this analysis. Original trained files and protocols remain unchanged.</p></main></html>''')
    print('Exported complete classical comparison report')


if __name__=='__main__':main()
