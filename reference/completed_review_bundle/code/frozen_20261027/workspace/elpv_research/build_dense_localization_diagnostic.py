"""Plot all source-selected systems without hiding the small-annotation failures."""
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,now

FOLDER=ROOT/'artifacts/publication_20261017/dense_source_localization_diagnostic_v1'
NAMES={'deeplabv3plus_full':'Adapted CNN', 'dinov3_dense_frozen':'DINOv3 frozen',
       'dinov3_dense_lora':'DINOv3 LoRA', 'dinov3_dense_full':'DINOv3 full'}
COLORS=['#667e87','#4d9caa','#7955a8','#1a6175']
AREA=['positive_le_0.5pct','positive_0.5_to_2pct','positive_gt_2pct']


def main():
    result=read_json(FOLDER/'results.json');plan=read_json(FOLDER/'plan.json')
    if result['plan_sha256']!=sha256(FOLDER/'plan.json') or plan['analysis_code_sha256']!=sha256(ROOT/'analyze_dense_source_localization.py'):
        raise RuntimeError('Source diagnostic evidence differs')
    families=result['families'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(12.2,4.7),sharex=True,sharey=True)
    for ax,family,color in zip(axes,families,COLORS):
        fits=family['fits'];rows=[r for r in fits[0]['per_image'] if r['target']]
        x=np.array([r['area_fraction']*100 for r in rows]);ys=[]
        for fit in fits:
            by={r['id']:r for r in fit['per_image']}
            ys.append([by[r['id']]['native_Dice'] for r in rows])
        ax.scatter(x,np.mean(ys,axis=0),c=color,s=20,alpha=.7,edgecolor='white',lw=.35,zorder=3)
        ax.set_xscale('log');ax.set_xlim(10**np.floor(np.log10(x.min())),100);ax.set_ylim(-.035,1.035)
        for value in [.5,2]:ax.axvline(value,c='#b3c1c6',ls='--',lw=.8,zorder=0)
        ax.set_title(NAMES[family['family']],loc='left',fontsize=11)
        ax.set_xlabel('Annotated crack / known pixels (%)',fontsize=8.8)
        ax.grid(axis='y',alpha=.15);ax.set_yticks([0,.25,.5,.75,1.])
    axes[0].set_ylabel('Per-image mean Dice over 3 fitted seeds')
    fig.suptitle('Native annotation footprint and localization overlap',x=.08,ha='left',fontsize=14,color='#193e48',y=.97)
    fig.subplots_adjust(left=.08,right=.985,top=.84,bottom=.28,wspace=.15)
    fig.text(.08,.12,'All 58 crack-positive UCF validation images in every panel. Dashed boundaries: 0.5% and 2% of known pixels.',fontsize=9,color='#526f79')
    fig.text(.08,.045,'These are source-selected checkpoint metrics, not an ensemble or independent test. Annotation footprint is not physical severity.\n'
             'No positive or failure is omitted; the same fixed pixel-score threshold 0.5 applies to these learned segmentation systems.',fontsize=8.7,color='#526f79',linespacing=1.5)
    for ext in ['png','pdf','svg']:fig.savefig(FOLDER/f'annotation_footprint_and_dice.{ext}',dpi=230,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    widths=.18;positions=np.arange(3)
    supports=[next(r['images'] for r in families[0]['fits'][0]['subgroups'] if r['subgroup']==s) for s in AREA]
    for index,(family,color) in enumerate(zip(families,COLORS)):
        for ax,key in zip(axes,['source_image_recall','image_detected_with_zero_overlap_fraction']):
            values=[[next(r[key] for r in fit['subgroups'] if r['subgroup']==s) for fit in family['fits']] for s in AREA]
            if any(any(v is None for v in subset) for subset in values):raise RuntimeError('Do not plot absent strata as zero')
            means=np.mean(values,axis=1);sd=np.std(values,axis=1,ddof=1)
            offset=positions+(index-1.5)*widths
            ax.bar(offset,means,width=widths*.9,color=color,label=NAMES[family['family']])
            ax.errorbar(offset,means,yerr=sd,fmt='none',color='#193e48',capsize=2,lw=.8)
            for center,subset in zip(offset,values):ax.scatter(np.full(3,center),subset,s=9,color='#193e48',alpha=.35,zorder=5)
    axes[0].set_title('a  Image-level crack recall',loc='left',fontsize=11)
    axes[1].set_title('b  Image detected, zero crack-pixel overlap',loc='left',fontsize=11)
    for ax in axes:
        ax.set_ylim(0,1.15);ax.set_yticks([0,.25,.5,.75,1.]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.set_xticks(positions,[f'≤0.5%\n(n={supports[0]})',f'0.5–2%\n(n={supports[1]})',f'>2%\n(n={supports[2]})'])
        ax.set_xlabel('Annotated crack fraction of known native pixels')
    axes[0].set_ylabel('Fraction of crack-positive images')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,frameon=False,fontsize=8.5,loc='lower left',bbox_to_anchor=(.085,.105))
    fig.subplots_adjust(left=.085,right=.985,top=.88,bottom=.31,wspace=.18)
    fig.text(.085,.025,'Mean ± sample SD over the same 3 fitted source-selected seeds; points retain individual values. Source validation also selects image thresholds.\n'
             'Zero overlap means native Dice = 0 at pixel threshold 0.5; it does not necessarily mean no predicted positive pixel anywhere.',fontsize=8.5,color='#526f79',linespacing=1.5)
    for ext in ['png','pdf','svg']:fig.savefig(FOLDER/f'image_detection_localization_disconnect.{ext}',dpi=230,facecolor='white')
    plt.close(fig)
    table=[];summaries=[]
    for family in families:
        row={'family':family['family'],'pooled_native_pixel_AP':float(np.mean([f['pooled_native_pixel_AP'] for f in family['fits']]))}
        for group in ['all_positives']+AREA:
            values=[next(r for r in fit['subgroups'] if r['subgroup']==group) for fit in family['fits']]
            row[group]={'images':values[0]['images'],'mean':{key:(float(np.mean([v[key] for v in values])) if values[0][key] is not None else None)
                for key in ['source_image_recall','mean_positive_image_Dice','mean_positive_image_AP','zero_overlap_fraction','image_detected_with_zero_overlap_fraction']}}
        summaries.append(row)
        for group in ['all_positives']+AREA:
            v=row[group];table.append('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in [NAMES[row['family']],group,v['images']]+
                [f"{v['mean'][key]:.3f}" if v['mean'][key] is not None else 'Not estimable' for key in ['source_image_recall','mean_positive_image_Dice','mean_positive_image_AP','zero_overlap_fraction','image_detected_with_zero_overlap_fraction']])+'</tr>')
    save_json(FOLDER/'figure_source.json',{'at':now(),'results_sha256':sha256(FOLDER/'results.json'),
        'figure_code_sha256':sha256(__file__),'summaries':summaries,
        'limits':'Post-hoc source-validation diagnostic. Pixel pooled AP and macro positive-image AP use different populations/weighting; neither substitutes for the other.'})
    (FOLDER/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Image detection versus native localization</title><style>body{margin:0;background:#f3f6f7;color:#193e48;font:16px/1.6 system-ui}main{max-width:1180px;background:white;margin:40px auto;padding:28px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #dce5e7;text-align:left;padding:8px}a{color:#236c81}.small{font-size:13px;color:#526f79}.scroll{overflow:auto}</style><main>
<h1>Image detection does not guarantee correct crack overlap</h1>
<p>This retrospective diagnostic retains every selected source system, all three fitted seeds and all144 UCF validation images. The positive footprint groups contain7,9 and42 images. The same validation set selects rates, checkpoints and image thresholds, so these are descriptive source-development findings.</p>
<img src="annotation_footprint_and_dice.png" alt="Every positive image: annotation footprint versus mean native Dice across the three fitted seeds">
<img src="image_detection_localization_disconnect.png" alt="All systems: source image recall and image detection with zero true crack overlap in each annotation-footprint group">
<p>The selected full-adaptation system has pooled native pixel AP0.736, but in the7 smallest-footprint images the mean fraction with zero overlap at pixel cutoff0.5 is0.905. LoRA has mean zero-overlap fraction0.952 in this group; the adapted CNN has0.619. These are means over the same images under three fitted seeds, not21 independent small-crack samples.</p>
<p>All four systems have mean source image recall0.966 across58 positives, partly imposed by the source high-recall threshold selection. This source construction cannot establish target recall. Some detected images still have no correct pixel overlap. Zero Dice can arise from missed or misplaced predictions; it does not prove a physical failure mechanism.</p>
<div class="scroll"><table><tr><th>System</th><th>Footprint group</th><th>Images</th><th>Image recall</th><th>Positive Dice</th><th>Macro positive-image AP</th><th>Zero overlap</th><th>Detected + zero overlap</th></tr>'''+''.join(table)+'''</table></div>
<p class="small">Macro positive-image AP averages AP across positive images only. Pooled pixel AP ranks pixels over all known native pixels, including negative images; they are distinct endpoints with different weighting. Small footprint also co-varies with population and annotation geometry, so area does not identify a cause or measure electrical severity. No independent interval, target fitting or target threshold is asserted here.</p>
<p><a href="all_seed_subgroups.csv">All60 seed/subgroup records</a> · <a href="results.json">All per-image outcomes and original source bindings</a> · <a href="annotation_footprint_and_dice.pdf">Footprint figure PDF</a> · <a href="image_detection_localization_disconnect.pdf">Detection/overlap figure PDF</a> · <a href="plan.json">Retrospective diagnostic plan</a></p></main></html>''')
    print('Exported full source localization diagnostic; all58 positives and60 seed/subgroup records retained',flush=True)


if __name__=='__main__':main()
