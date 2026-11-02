"""Observed source annotation support, not model-score attribution or new fitting."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now

BASE=ROOT/'artifacts/publication_20261017'
OUT=BASE/'native_annotation_support_v1'


def annotation_rows(path,cohort):
    run=read_json(path);native=run['validation_localization']['per_image']
    rows=[]
    for row in native:
        known,positive=int(row['known_pixels']),int(row['crack_pixels'])
        if not 0<=positive<=known or known<=0 or bool(positive)!=bool(row['crack_positive']):
            raise RuntimeError('Native known/crack counts or image truth disagree')
        source='UCF'
        if cohort=='SCDD':
            source=row['id'].split('/',1)[1].split('_',1)[0]
            if source not in ['ARTS','CFVS','CSIR'] or not row['acquisition_family'].startswith(source+'_'):
                raise RuntimeError('Unexpected source/capture prefix; do not invent an acquisition source')
        rows.append(dict(id=row['id'],cohort=cohort,provided_source=source,
            capture_family=row['acquisition_family'],known_pixels=known,crack_pixels=positive,
            crack_positive=bool(positive)))
    rows.sort(key=lambda r:r['id'])
    if len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!=set(run['validation_ids']):
        raise RuntimeError('Native annotation identities differ from validation cohort')
    return rows


def support(rows):
    positive=[r for r in rows if r['crack_positive']]
    known=sum(r['known_pixels'] for r in rows);pixels=sum(r['crack_pixels'] for r in rows)
    ordered=sorted(positive,key=lambda r:(-r['crack_pixels'],r['id']))
    if not ordered or pixels<=0:raise RuntimeError('Require a positive source cohort')
    k=math.ceil(.1*len(ordered))
    return dict(images=len(rows),positive_images=len(positive),provided_capture_families=len({r['capture_family'] for r in rows}),
        known_pixels=known,annotated_crack_pixels=pixels,image_positive_fraction=len(positive)/len(rows),
        constant_pixel_score_AP=pixels/known,top10pct_image_count=k,
        top10pct_positive_image_crack_pixel_share=sum(r['crack_pixels'] for r in ordered[:k])/pixels,
        sorted_positive_image_ids=[r['id'] for r in ordered],
        cumulative_positive_image_fraction=[0]+list(np.arange(1,len(ordered)+1)/len(ordered)),
        cumulative_annotated_crack_pixel_fraction=[0]+list(np.cumsum([r['crack_pixels'] for r in ordered])/pixels))


def plot(cohorts,sources,folder):
    style={'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':'#8fa2aa','axes.labelcolor':'#294953',
        'xtick.color':'#3a555e','ytick.color':'#3a555e','pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(matplotlib.rcParamsDefault),plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(7.2,4.1))
        names=['ARTS','CFVS','CSIR'];y=np.arange(3)
        image=[100*sources[n]['positive_image_share_within_cohort'] for n in names]
        pixels=[100*sources[n]['crack_pixel_share_within_cohort'] for n in names]
        axes[0].barh(y-.17,image,height=.29,color='#286c7c',label='Crack-positive images')
        axes[0].barh(y+.17,pixels,height=.29,color='#bd8840',label='Annotated crack pixels')
        axes[0].set_yticks(y,names);axes[0].set_xlim(0,108);axes[0].invert_yaxis()
        for ys,values in [(y-.17,image),(y+.17,pixels)]:
            for yy,xx in zip(ys,values):axes[0].text(xx+1,yy,f'{xx:.1f}%',va='center',fontsize=7)
        axes[0].set_xticks([0,25,50,75,100]);axes[0].set_xlabel('Share within SCDD cohort (%)')
        axes[0].legend(loc='upper left',bbox_to_anchor=(-.07,-.22),frameon=False,fontsize=7.5)
        for name,color in [('SCDD','#286c7c'),('UCF','#795aaf')]:
            r=cohorts[name]
            axes[1].step(100*np.array(r['cumulative_positive_image_fraction']),100*np.array(r['cumulative_annotated_crack_pixel_fraction']),
                         where='post',lw=1.7,color=color,label=f"{name}: {r['positive_images']} positives")
        axes[1].plot([0,100],[0,100],ls='--',color='#9babad',lw=1,label='Uniform pixel support')
        axes[1].set_xlim(0,100);axes[1].set_ylim(0,103);axes[1].set_xticks([0,25,50,75,100]);axes[1].set_yticks([0,25,50,75,100])
        axes[1].set_xlabel('Crack-positive images, sorted by area (%)')
        axes[1].set_ylabel('Cumulative annotated crack pixels (%)')
        axes[1].legend(loc='lower right',frameon=False,fontsize=7)
        for ax,title in zip(axes,['a  Sources have different pixel support','b  A few masks can dominate pixel counts']):
            ax.set_title(title,loc='left',fontsize=9,pad=12)
            ax.grid(axis='x',color='#e6ecee',lw=.6);ax.set_axisbelow(True)
        fig.subplots_adjust(left=.08,right=.985,top=.86,bottom=.29,wspace=.42)
        fig.text(.08,.035,'Observed validation masks only. Pixel support is not AP attribution or physical crack severity.',fontsize=7.5,color='#56717b')
        for ext in ['png','pdf','svg']:fig.savefig(folder/f'annotation_support.{ext}',dpi=300,facecolor='white')
        plt.close(fig)


def main():
    if (BASE/'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Complete source annotation analysis before final model observation')
    OUT.mkdir(exist_ok=True)
    plan=dict(kind='Retrospective source annotation-support audit, independent of model scores',
        code_sha256=sha256(__file__),source_cohorts={'UCF':'All24 complete dense-calibration source-validation summaries',
            'SCDD':'Every model/seed in fully complete internal dense cells; same120 validation identities'},
        fields_used=['id','acquisition_family','known_pixels','crack_pixels','crack_positive','validation_ids'],
        source_names='SCDD author/file acquisition prefixes ARTS/CFVS/CSIR; not verified physical module identities. UCF remains provenance-conditional.',
        concentration_rule='Sort all positive images by annotated crack-pixel count, descending; ID breaks ties. Top ceil(10% of positives) and cumulative shares. No model-score selection.',
        constant_score_AP='Annotated crack pixels / all known canonical pixels. This is the constant-score AP baseline, not a performance result or AP contribution decomposition.',
        limits=['Native annotation area is not physical crack width, electrical severity or annotation time.',
            'Annotation concentration alone does not explain model rankings, isolate acquisition from annotation conventions or decompose pooled AP.',
            'Known-pixel totals include canonical normal/crack/other pixels and exclude task IGNORE regions.',
            'No bootstrap over pixels and no statistical interval from treating model copies as independent annotations.'],
        final_confirmation_access=False,raw_images_or_masks_read=False,new_fit_or_inference=False)
    plan_path=OUT/'plan.json'
    if plan_path.exists():
        if read_json(plan_path)!=plan:raise RuntimeError('Preserve/amend changed annotation-support analysis')
    else:save_json(plan_path,plan)
    paths=[('UCF',p) for p in sorted((BASE/'dense_development_v1/calibration').glob('*/run.json'))]
    if len(paths)!=24:raise RuntimeError('Require all24 dense source calibrations')
    cells=read_json(BASE/'dense_development_complete_cells.json')['complete_cells']
    for cell in cells:
        if cell['fold']!='internal':continue
        for binding in cell['identity']['bindings']:
            if sha256(ROOT/binding['run'])!=binding['sha256']:raise RuntimeError('Completed source run changed')
            paths.append(('SCDD',ROOT/binding['run']))
    cohorts={};bindings=[];verified={}
    for name,path in paths:
        rows=annotation_rows(path,name)
        if name in cohorts and cohorts[name]!=rows:raise RuntimeError('Native annotation counts/identities differ across matched models or seeds')
        cohorts[name]=rows;verified[name]=verified.get(name,0)+1
        bindings.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    if len(cohorts['UCF'])!=144 or len(cohorts['SCDD'])!=120:raise RuntimeError('Unexpected source cohort')
    identity=digest({'plan_sha256':sha256(plan_path),'bindings':bindings})[:20];folder=OUT/identity;folder.mkdir(exist_ok=True)
    result_path=folder/'results.json'
    if not result_path.exists():
        summaries={name:support(rows) for name,rows in cohorts.items()}
        source_rows={}
        for name in ['ARTS','CFVS','CSIR']:
            source_rows[name]=support([r for r in cohorts['SCDD'] if r['provided_source']==name])
            source_rows[name]['positive_image_share_within_cohort']=source_rows[name]['positive_images']/summaries['SCDD']['positive_images']
            source_rows[name]['crack_pixel_share_within_cohort']=source_rows[name]['annotated_crack_pixels']/summaries['SCDD']['annotated_crack_pixels']
        with (folder/'per_image_annotation_counts.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(cohorts['UCF'][0]));writer.writeheader()
            writer.writerows(cohorts['UCF']+cohorts['SCDD'])
        plot(summaries,source_rows,folder)
        save_json(result_path,{'at':now(),'identity':identity,'plan_sha256':sha256(plan_path),
            'input_bindings':bindings,'same_native_annotations_verified_in_run_summaries':verified,
            'cohorts':summaries,'SCDD_provided_source_support':source_rows,
            'figures':[{'path':str((folder/f'annotation_support.{ext}').relative_to(ROOT)),'sha256':sha256(folder/f'annotation_support.{ext}')} for ext in ['png','pdf','svg']],
            'scope':plan['kind'],'limits':plan['limits'],'raw_images_or_masks_read':False,'final_confirmation_access':False})
    result=read_json(result_path)
    save_json(OUT/'latest.json',{'at':now(),'results':str(result_path.relative_to(ROOT)),'sha256':sha256(result_path),'verified_run_summaries':verified})
    print('Annotation support:',verified,'identical source annotation sets; no model outcomes used')
    for name,r in result['cohorts'].items():print(name,{k:r[k] for k in ['images','positive_images','known_pixels','annotated_crack_pixels','constant_pixel_score_AP','top10pct_image_count','top10pct_positive_image_crack_pixel_share']})
    for name,r in result['SCDD_provided_source_support'].items():print(name,{k:r[k] for k in ['positive_images','positive_image_share_within_cohort','crack_pixel_share_within_cohort']})


if __name__=='__main__':main()
