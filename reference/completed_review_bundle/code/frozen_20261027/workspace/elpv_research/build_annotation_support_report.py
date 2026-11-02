"""Present the saved source-annotation audit without changing its analysis."""
import html
from pathlib import Path

from elresearch.common import ROOT,read_json,save_json,sha256,now

BASE=ROOT/'artifacts/publication_20261017/native_annotation_support_v1'


def main():
    latest=read_json(BASE/'latest.json');path=ROOT/latest['results'];r=read_json(path)
    if sha256(path)!=latest['sha256']:raise RuntimeError('Annotation-support result changed')
    plan=BASE/'plan.json'
    if r['plan_sha256']!=sha256(plan) or read_json(plan)['code_sha256']!=sha256(ROOT/'analyze_native_annotation_support.py'):
        raise RuntimeError('Original annotation-support definition changed')
    for binding in r['input_bindings']:
        if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Original annotation-count source changed')
    for f in r['figures']:
        if sha256(ROOT/f['path'])!=f['sha256']:raise RuntimeError('Annotation-support plot changed')
    u,s=r['cohorts']['UCF'],r['cohorts']['SCDD'];sources=r['SCDD_provided_source_support']
    latex=[r'\subsection{Native annotation support and metric weighting}',
        'A retrospective audit uses the original native known/crack pixel counts in saved source-validation summaries. The annotation identities and counts agree across all included matched model/seed evaluations of each cohort. Those repeated outputs are checks of the same annotations, not independent annotation replications. The two cohorts contain 144 UCF and 120 SCDD images. Canonical task-ignored pixels are excluded from the denominator.',
        '',r'\begin{table}[htbp]\centering\small',
        r'\caption{Observed source-validation annotation support. The last column is annotated crack pixels divided by all known canonical pixels: exactly the AP of a constant pixel score, not an experimentally trained baseline. UCF retains its file-level provenance limitation.}',
        r'\begin{tabular}{lrrrr}\toprule',r'Cohort & Images & Crack-positive & Known pixels & Constant-score pixel AP\\\midrule']
    for name,c in [('UCF',u),('SCDD',s)]:
        latex.append(f"{name} & {c['images']} & {c['positive_images']} & {c['known_pixels']:,} & {c['constant_pixel_score_AP']:.6f}"+r'\\')
    latex += [r'\bottomrule\end{tabular}',r'\end{table}',
        f"The cohorts have {u['annotated_crack_pixels']:,} and {s['annotated_crack_pixels']:,} annotated crack pixels, respectively. In SCDD, CFVS contributes {100*sources['CFVS']['positive_image_share_within_cohort']:.1f}\\% of positive images but {100*sources['CFVS']['crack_pixel_share_within_cohort']:.1f}\\% of annotated crack pixels. ARTS contributes {100*sources['ARTS']['positive_image_share_within_cohort']:.1f}\\% of positive images and {100*sources['ARTS']['crack_pixel_share_within_cohort']:.1f}\\% of crack pixels; CSIR contributes {100*sources['CSIR']['positive_image_share_within_cohort']:.1f}\\% and {100*sources['CSIR']['crack_pixel_share_within_cohort']:.1f}\\%, respectively. The six largest positive SCDD masks account for {100*s['top10pct_positive_image_crack_pixel_share']:.1f}\\% of its annotated crack pixels, compared with {100*u['top10pct_positive_image_crack_pixel_share']:.1f}\\% for the six largest UCF masks.",
        'This observed support distribution motivates reporting both pooled native pixel AP and equally weighted positive-image Dice. Positive-pixel shares do not decompose AP, which also depends on score ordering and noncrack pixels. The counts do not measure physical crack size, electrical severity or annotation time, and do not isolate annotation conventions from acquisition/content. Different source fits, labels and pixel prevalences prevent ranking the two cohorts\' AP values as the same experiment.',
        '']
    fragment=BASE/'manuscript_fragment.tex';fragment.write_text('\n'.join(latex)+'\n')
    escape=html.escape
    table=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [name,c['images'],c['positive_images'],f"{c['known_pixels']:,}",f"{c['annotated_crack_pixels']:,}",f"{c['constant_pixel_score_AP']:.6f}"])+'</tr>' for name,c in [('UCF',u),('SCDD',s)])
    image=str((ROOT/r['figures'][0]['path']).relative_to(BASE))
    csv_path=str(path.parent.relative_to(BASE))+'/per_image_annotation_counts.csv'
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Annotation support and pixel metric weighting</title>
<style>body{font:16px/1.55 system-ui;color:#25434e;max-width:1180px;margin:32px auto;padding:0 24px}h1{line-height:1.15}img{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #d4dfe2}th{background:#eef4f5}a{color:#286c7c}.note{background:#f2f5f6;padding:16px;border-left:4px solid #bd8840}</style>
<h1>Which annotations enter a pooled pixel metric?</h1>
<p class="note">These are observed source-validation annotations, not new predictions or independent tests. Every included matched model/seed has identical native annotation counts and identities within its cohort. Repeated evaluation copies do not create new annotation samples.</p>
<img src="'''+image+'''" alt="All three SCDD source contributions and cumulative annotated crack-pixel support over every positive image in each source-validation cohort">
<table><tr><th>Cohort</th><th>Images</th><th>Crack-positive</th><th>Known pixels</th><th>Annotated crack pixels</th><th>Constant-score pixel AP</th></tr>'''+table+'''</table>
<p>The constant-score AP is the annotated crack-pixel fraction among known pixels. It supplies prevalence context, not a trained method or a normalized score for ranking datasets. UCF per-file real/simulated provenance remains unresolved; SCDD uses the audited real-image source cohort.</p>
<p>In SCDD, CFVS supplies 59.3% of positive images but 86.7% of annotated crack pixels. The six largest positive masks supply 62.8% of all annotated crack pixels. These figures motivate checking small cases and equally weighted positive-image overlap alongside pooled AP.</p>
<p>Pixel support is not AP attribution: AP depends on all score rankings and negative pixels as well. No physical severity, annotation effort, causal explanation of a model difference or new independent replication is inferred. Only already observed validation summaries are read; no raw image/mask, pretrained model, final prediction or new fit is used.</p>
<p><a href="'''+csv_path+'''">Every source-validation image count</a> · <a href="'''+str(path.relative_to(BASE))+'''">Exact source bindings and both complete cohorts</a> · <a href="plan.json">Recorded retrospective analysis</a> · <a href="'''+image.replace('.png','.pdf')+'''">Vector PDF</a></p></html>'''
    (BASE/'report.html').write_text(body)
    save_json(BASE/'report_inputs.json',{'at':now(),'analysis_results':latest,
        'analysis_plan_sha256':sha256(plan),'report_builder_sha256':sha256(__file__),
        'manuscript_fragment':{'path':str(fragment.relative_to(ROOT)),'sha256':sha256(fragment)},
        'figure':r['figures'][0],'new_model_result':False,'final_confirmation_access':False})
    print('Source annotation-support report and manuscript fragment exported')


if __name__=='__main__':main()
