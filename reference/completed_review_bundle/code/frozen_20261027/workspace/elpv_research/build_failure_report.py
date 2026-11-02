"""Readable development-only failure evidence; do not rerun model inference."""
from pathlib import Path
import html
import numpy as np

from elresearch.common import ROOT, read_json, save_json, sha256, now

BASE=ROOT/'artifacts/publication_20261017'
FOLDER=BASE/'image_failure_analysis_v1'
FAMILIES=[('dinov3_frozen','DINOv3-S frozen'),('dinov3_lora','DINOv3-S LoRA'),('dinov3_full','DINOv3-S full'),
    ('resnet18_full','ResNet-18 full'),('resnet50_full','ResNet-50 full'),('efficientnet_b0_full','EfficientNet-B0 full')]


def main():
    source=FOLDER/'results.json'; data=read_json(source); rows=data['subgroups']; area=[]; false_positive=[]; calibration=[]
    for acquisition in ['ARTS','CFVS','CSIR']:
        cells=[]
        for subgroup in ['positive_le_0.5pct','positive_0.5_to_2pct','positive_gt_2pct']:
            r=next(r for r in rows if r['held_source']==acquisition and r['training_fraction']==.2 and r['family']=='dinov3_lora' and r['subgroup']==subgroup)
            cells.append('0 · not estimable' if not r['images'] else f"{r['images']} · recall {r['mean_correct_fraction']:.3f}")
        area.append('<tr><th>'+acquisition+'</th>'+''.join('<td>'+value+'</td>' for value in cells)+'</tr>')
    for budget in [.2,1.]:
        table=[]
        for family,label in FAMILIES:
            cells=[]
            for acquisition in ['ARTS','CFVS','CSIR']:
                r=next(r for r in rows if r['held_source']==acquisition and r['training_fraction']==budget and r['family']==family and r['subgroup']=='all_negatives')
                cells.append(f"{1-r['mean_correct_fraction']:.3f} ± {r['fitted_seed_sample_sd']:.3f}")
            table.append('<tr><th>'+label+'</th>'+''.join('<td>'+value+'</td>' for value in cells)+'</tr>')
        false_positive.append(f'<h3>{int(budget*100)}% source-training labels</h3><table><tr><th>System</th><th>ARTS · n=106 negatives</th><th>CFVS · n=59</th><th>CSIR · n=96</th></tr>'+''.join(table)+'</table>')
        for family,label in FAMILIES:
            for acquisition in ['ARTS','CFVS','CSIR']:
                records=[r for r in data['calibration'] if r['family']==family and r['training_fraction']==budget and r['held_source']==acquisition]
                raw=float(np.mean([r['raw']['brier'] for r in records])); calibrated=float(np.mean([r['source_temperature']['brier'] for r in records]))
                nll_raw=float(np.mean([r['raw']['negative_log_likelihood'] for r in records]));nll_cal=float(np.mean([r['source_temperature']['negative_log_likelihood'] for r in records]))
                calibration.append({'system':label,'source':acquisition,'budget':budget,'raw_brier':raw,'source_T_brier':calibrated,
                    'brier_difference_source_T_minus_raw':calibrated-raw,'raw_NLL':nll_raw,'source_T_NLL':nll_cal})
    lora=[r for r in calibration if r['system']=='DINOv3-S LoRA']
    calibration_rows=''.join(f"<tr><th>{r['source']}</th><td>{int(r['budget']*100)}%</td><td>{r['raw_brier']:.4f}</td><td>{r['source_T_brier']:.4f}</td><td>{r['brier_difference_source_T_minus_raw']:+.4f}</td></tr>" for r in lora)
    all_rows=''.join(f"<tr><th>{html.escape(r['system'])}</th><td>{r['source']}</td><td>{int(r['budget']*100)}%</td><td>{r['raw_brier']:.4f}</td><td>{r['source_T_brier']:.4f}</td><td>{r['brier_difference_source_T_minus_raw']:+.4f}</td></tr>" for r in calibration)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EL development failure analysis</title><style>body{margin:36px auto;padding:0 24px;max-width:1120px;color:#193d49;background:#f4f7f8;font:16px/1.6 system-ui}h1{line-height:1.2}h2{font-size:22px}section{background:white;border:1px solid #dce6e9;border-radius:10px;margin:22px 0;padding:24px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;border-bottom:1px solid #dfe8eb;text-align:left}img{display:block;width:100%}.small{font-size:13px;color:#597681}a{color:#176c80}.notice{background:#f6ebd6;padding:12px 16px;border-radius:6px}.scroll{overflow:auto}</style>
<h1>What the completed image study misses</h1><p class="notice">Retrospective SCDD development analysis. All six systems and all three seeds are retained. No independent confirmation scores, target-fitted thresholds or new training features.</p>
<section><h2>Annotation footprint differs across acquisition sources</h2>
<p>All45 ARTS crack-positive development images lie in the smallest annotated-area bin. CFVS and CSIR also contain larger annotated regions. Area therefore co-varies with acquisition source, field of view and annotation extent; it is not a measurement of physical crack width or electrical severity.</p>
<div class="scroll"><table><tr><th>Held source</th><th>Crack ≤0.5% of known pixels</th><th>0.5–2%</th><th>&gt;2%</th></tr>AREA</table></div>
<p class="small">Cell: image count and LoRA recall, averaged over three fitted seeds at20% source-training labels. Empty bins are not estimable. The bins are fixed descriptive thresholds, not learned cutoffs. Within the broad lowest bin, sources can still differ in area distributions and other factors; these counts do not identify the cause of source performance differences.</p></section>
<section><h2>High sensitivity can require many false alarms</h2><p>False-positive rate among crack-negative images, at each model’s own source-validation high-recall threshold. Lower is better. This differs from the fraction of all images sent for review.</p>FALSE_POSITIVE
<p class="small">Mean ± fitted-seed sample SD, not a confidence interval. Negative means no annotated crack and can include other defects. Source95% recall is not a guaranteed target rate.</p></section>
<section><h2>Source probability calibration need not improve the target</h2>
<p>LoRA source-temperature scaling reduces mean Brier loss in five of these six development cells, but increases it on CFVS at20% labels. This is a source-fitted transformation; target labels are used only to evaluate it.</p>
<table><tr><th>Source</th><th>Training budget</th><th>Raw Brier</th><th>Source-T Brier</th><th>Difference</th></tr>CALIBRATION</table>
<p class="small">Brier loss is lower when probabilities are more accurate; positive differences indicate worse target calibration. Means condition on the three fitted seeds. No uncertainty or universal calibration benefit is claimed. A positive temperature does not change ranking.</p>
<details><summary>All six systems and both budgets</summary><div class="scroll"><table><tr><th>System</th><th>Source</th><th>Budget</th><th>Raw Brier</th><th>Source-T Brier</th><th>Difference</th></tr>ALL_CALIBRATION</table></div></details></section>
<section><h2>Selected failures shown against real native annotations</h2><img src="development_failure_examples.png" alt="Six selected development images with native crack, structure, other-defect and excluded annotations; incorrect-seed counts for every model">
<p>Frozen / LoRA / Full denote DINOv3-S; R18, R50 and Eff denote the CNNs. The CSIR positive is missed by one LoRA seed; it is not presented as a three-seed failure. The CFVS negative is flagged by all LoRA seeds and none of the EfficientNet seeds. The selected CSIR negative is flagged by every seed of every system.</p>
<p class="small">Images were selected using the recorded maximum-LoRA-error rule with annotation-area/structure tie breaks; this is not a representative sample. Ground-truth overlays are not predictions or attention maps. Structural co-occurrence cannot prove which image feature caused a decision. <a href="development_failure_examples.pdf">Vector PDF</a> · <a href="plan.json">Descriptive plan and export amendment</a></p></section>
<section><h2>Traceable inputs</h2><p><a href="results.json">Native characteristics, all seed results, cases and hashes</a> · <a href="all_seed_subgroup_counts.csv">All972 seed/subgroup records</a> · <a href="../publication_progress.html">Main research report</a></p>
<p class="small">Full471 observed development masks define evaluation strata only; masks do not become inputs or supervision to the image-only models. Empty subgroups stay null. The117 SCDD test images and other confirmation cohorts remain sealed.</p></section></html>'''
    for token,value in [('ALL_CALIBRATION',all_rows),('FALSE_POSITIVE',''.join(false_positive)),('CALIBRATION',calibration_rows),('AREA',''.join(area))]:page=page.replace(token,value)
    (FOLDER/'report.html').write_text(page)
    save_json(FOLDER/'report_inputs.json',{'at':now(),'results_sha256':sha256(source),'builder_sha256':sha256(__file__),
        'calibration_summary':calibration,'scope':'Retrospective development point summaries; no new model fitting or confirmation opening.'})
    print('Failure report generated;',len(calibration),'calibration system/cell summaries')


if __name__=='__main__':main()
