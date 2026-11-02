"""Evidence status and exportable data-design figures; never fabricate missing results."""
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT,now,read_json,save_json,sha256


BASE=ROOT/'artifacts/publication_20261017'


def figures():
    audit=read_json(BASE/'scdd_data_audit.json');design=read_json(BASE/'scdd_development_v1/design.json')
    output=BASE/'figures';output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':'#82969c','axes.labelcolor':'#243d46',
        'xtick.color':'#334e58','ytick.color':'#334e58','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),gridspec_kw={'width_ratios':[1,1.2]})
    sources=['ARTS','CFVS','CSIR'];positions=np.arange(3);bottom=np.zeros(3)
    for partition,color,label in [('train','#246779','Training'),('validation','#8eb9c2','Validation'),
                                   ('test','#d5dee1','Sealed test')]:
        values=np.array([audit['source_counts'][s][partition]['images'] for s in sources])
        axes[0].barh(positions,values,left=bottom,color=color,label=label,height=.6,
                     edgecolor='white',linewidth=1)
        for y,x,left in zip(positions,values,bottom):
            axes[0].text(left+x/2,y,str(x),va='center',ha='center',color='white' if partition=='train' else '#163b48',fontsize=10)
        bottom+=values
    axes[0].set_yticks(positions,['ARTsolar','CFV','CSIR']);axes[0].invert_yaxis()
    axes[0].set_xlabel('Original, non-augmented images')
    axes[0].set_title('a  Real-image cohort by acquisition source',loc='left',fontsize=11,pad=15)
    axes[0].legend(frameon=False,loc='lower left',bbox_to_anchor=(-.04,-.32),ncol=3,fontsize=9)
    axes[0].set_xlim(0,235)
    internal=design['split_manifest']['folds'][0]
    budgets=[r for r in internal['annotation_budget_counts'] if r['seed']==11]
    x=np.arange(len(budgets));training=np.array([r['training_images'] for r in budgets]);validation=120
    axes[1].bar(x,training,color='#246779',width=.58,label='Training labels')
    axes[1].bar(x,[validation]*4,bottom=training,color='#8eb9c2',width=.58,label='Validation labels')
    for i,n in enumerate(training):
        axes[1].text(i,n+validation+9,str(n+validation),ha='center',fontsize=10,fontweight='bold',color='#183c47')
        if n>30:axes[1].text(i,n/2,str(n),ha='center',va='center',color='white',fontsize=10)
    axes[1].set_xticks(x,['5%','10%','20%','100%']);axes[1].set_xlabel('Fraction of the 351-image training pool')
    axes[1].set_ylabel('SCDD labels used, including validation')
    axes[1].set_ylim(0,530);axes[1].set_title('b  Label availability includes validation',loc='left',fontsize=11,pad=15)
    axes[1].legend(frameon=False,loc='lower left',bbox_to_anchor=(-.04,-.32),ncol=2,fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.86,bottom=.27,wspace=.43)
    fig.text(.09,.03,'588 real images; 471 development / 117 sealed test. Simulated label budgets; physical module IDs unconfirmed.',
             fontsize=8.5,color='#56717b')
    for ext in ['png','pdf','svg']:fig.savefig(output/f'data_and_annotation_design.{ext}',dpi=250,facecolor='white')
    plt.close(fig)
    save_json(output/'data_and_annotation_design_source.json',{'created_at':now(),
        'scdd_audit_sha256':sha256(BASE/'scdd_data_audit.json'),'design_sha256':sha256(BASE/'scdd_development_v1/design.json'),
        'counts':audit['source_counts'],'internal_annotation_budgets_seed11':budgets,
        'meaning':'Data and protocol figure, not predictive performance.'})
    selection=BASE/'calibration_v1/selected_hyperparameters.json'
    if selection.exists():
        chosen=read_json(selection);fig,ax=plt.subplots(figsize=(9.2,4.5))
        labels={'dinov3_frozen':'DINOv3-S · frozen + linear',
            'dinov3_lora':'DINOv3-S · LoRA + linear','dinov3_full':'DINOv3-S · full + linear',
            'resnet18_full':'ResNet-18 · full + linear','resnet50_full':'ResNet-50 · full + linear',
            'efficientnet_b0_full':'EfficientNet-B0 · full + linear'}
        for i,row in enumerate(chosen['selected']):
            family=row['specification']['family'];color='#7753a5' if family.startswith('dinov3') else '#246779'
            values=np.array([r['auroc'] for r in row['runs']]);mean=values.mean();sd=values.std(ddof=1)
            ax.scatter(values,np.full(len(values),i)+np.linspace(-.12,.12,len(values)),s=22,color=color,alpha=.45,zorder=2)
            ax.errorbar(mean,i,xerr=sd,fmt='o',color=color,ms=6,capsize=4,lw=1.6,zorder=3)
            ax.text(1.025,i,f'{mean:.3f} ± {sd:.3f}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=9,color='#334e58')
        ax.set_yticks(range(len(chosen['selected'])),[labels[r['specification']['family']] for r in chosen['selected']])
        ax.invert_yaxis();ax.set_xlim(.88,1.);ax.set_ylim(5.6,-.7)
        ax.set_xlabel('Image AUROC on reused UCF development validation')
        ax.set_title('Selected source-calibration models',loc='left',fontsize=13,pad=16)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#e8edef',lw=.8);ax.set_axisbelow(True)
        fig.subplots_adjust(left=.32,right=.82,top=.85,bottom=.24)
        fig.text(.055,.06,'Dots: three seeds; error bars: sample SD, not confidence intervals. Learning rates selected on this same validation set.',fontsize=8,color='#56717b')
        fig.text(.055,.02,'Image labels only; UCF provenance provisional. These scores do not establish external superiority.',fontsize=8,color='#56717b')
        for ext in ['png','pdf','svg']:fig.savefig(output/f'source_calibration_summary.{ext}',dpi=250,facecolor='white')
        plt.close(fig)
        save_json(output/'source_calibration_summary_source.json',{'at':now(),
            'hyperparameter_selection_sha256':sha256(selection),'selected':chosen['selected'],
            'interpretation':'Source-only selected development calibration, not independent confirmation; sample SD is not a confidence interval.'})


def main():
    figures()
    from build_publication_development_figures import main as development_figures
    development = development_figures()
    if development:
        from analyze_source_shortcuts import main as source_diagnostics
        source_diagnostics()
    from build_held_source_progress import main as held_source_figures
    held=held_source_figures()
    from build_held_source_matrix import main as held_source_matrix
    held_source_matrix()
    if len(held['complete_cells'])==6:
        from build_inspection_tradeoff import main as inspection_tradeoff
        inspection_tradeoff()
    if (BASE/'held_source_paired_uncertainty.json').exists():
        from build_held_source_contrasts import main as conditional_contrasts
        conditional_contrasts()
    if (BASE/'classical_development_analysis_v1/results.json').exists():
        from build_classical_report import main as classical_report
        classical_report()
    dense_cells=None
    if (BASE/'dense_development_v1/selected_hyperparameters.json').exists():
        from build_dense_development_progress import main as dense_progress
        dense_cells=dense_progress()
        if dense_cells['complete_cells']:
            from build_dense_development_gallery import main as dense_gallery
            dense_gallery()
            from build_dense_transfer_readout import main as dense_transfer_readout
            dense_transfer_readout()
            from build_dense_operating_readout import main as dense_operating_readout
            dense_operating_readout()
    dense_uncertainty_path=BASE/'dense_development_uncertainty_v1/index.json'
    if dense_uncertainty_path.exists():
        from build_dense_uncertainty_report import main as dense_uncertainty_report
        dense_uncertainty_report()
        held_dense = [c for c in read_json(dense_uncertainty_path)['complete_cells'] if c['fold'] != 'internal']
        if len(held_dense) == 6:
            from build_dense_transfer_summary import main as dense_transfer_summary
            dense_transfer_summary()
    if read_json(BASE/'ucf_reference_transfer_v1/status.json')['state']=='complete':
        from build_ucf_reference_readout import main as ucf_reference_readout
        ucf_reference_readout()
    calibration_done=(BASE/'calibration_v1/selected_hyperparameters.json').exists()
    evidence={'updated_at':now(),'submission_ready':False,
        'working_question':'How do frozen, low-rank and full adaptation compare with fully trained CNNs under limited labels and acquisition shift?',
        'completed':['27 advisor head/adaptation pilot runs, three seeds, exact checkpoint reload verified',
                     'Real SCDD cohort: 588 original images, D4 duplicate checks, three acquisition sources',
                     'UCF supplemental confirmation acquisition: 507 images after source-family and annotation exclusions',
                     'Official pretrained DINOv3, ResNet18, ResNet50 and EfficientNet-B0 forward/backward checks',
                     str(read_json(BASE/'implementation_verification_latest.json')['passed'])+' full-suite implementation checks passed; these verify software, not research superiority',
                     '1,045 previously observed UCF/SCDD source images rechecked for exact-D4 identity; no confirmation scoring',
                     'All four internal image-training-label budgets complete for all six neural systems and all three paired seeds',
                     '15,963-image whole-frame similarity audit;17 PVEL near-copy pairs excluded from prospective primary analysis; unchanged full pool retained',
                     'Eight retrospective possible-related-family sensitivities complete; fitted models and point differences unchanged',
                     'Official Small/Base and ImageNet ResNet18 CPU shape/pooling checks passed on an observed source image; no Base performance result'],
        'image_study':read_json(BASE/'scdd_development_v1/status.json'),
        'required_before_submission':['Complete and verify strong baseline, label-budget and acquisition-shift results',
            'Select a defensible contribution from development evidence; extend localization if supported by matched pixel-supervision experiments',
            'Freeze the final model(s), threshold(s) and confirmation analysis; then evaluate independent confirmation once',
            'Paired acquisition-family uncertainty, error analysis, calibration and actual efficiency measurements',
            'Update primary literature comparison, English manuscript, reproducibility package and data-use statements'],
        'limits':['Reused UCF pilot validation cannot establish independent superiority.',
            'UCF test-real status is an inference from the paper and release policy, not verified file-level provenance.',
            'Different pretrained corpora prevent a causal Transformer-versus-CNN claim.',
            '5% training labels plus 120 validation labels is not 5% total annotation.',
            'Class-stratified subsets simulate label availability; this is not a measured prospective annotation-cost saving. UCF source labels used for hyperparameter calibration are additional to SCDD counts.',
            'SCDD field of view includes neighboring cells; image labels do not measure central-cell electrical failure.'],
        'literature_update':'Primary 2026 papers already study DINOv3 industrial transfer and EL cross-domain diagnosis. These ingredients alone are not new. See literature_update.json for access and date limits.',
        'sealed':{'scdd_original_test_model_scores':117,'ucf_supplemental_model_scores':507,'elpv_course_test':'untouched',
                  'qualification':'Candidate-confirmation pixels reviewed for integrity. Model scores, fitting and thresholds remain source-only.'},
        'followup':'Hourly thread heartbeat created in Codex; reuse workers, repair failures with an audit trail, progress on evidence.'}
    save_json(BASE/'evidence_status.json',evidence)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DINOv3 PV — Publication Evidence</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f7;color:#183b46;font:16px/1.6 system-ui,sans-serif}
main{max-width:1120px;margin:48px auto;padding:0 28px}h1{font-size:34px;line-height:1.2;margin:10px 0}h2{font-size:22px;margin:0 0 14px}
.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#407c89}.lead{max-width:850px;color:#506c76}
.badge{display:inline-block;padding:5px 12px;border-radius:18px;background:#f9e9cf;color:#7c571e;font-size:13px;font-weight:600}
section{background:white;border:1px solid #dce5e8;border-radius:12px;padding:25px 28px;margin:22px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:#f4f8f9;border:1px solid #dbe7ea;border-radius:9px;padding:18px}
.small{font-size:13px;color:#607982}.number{font-size:29px;font-weight:650;color:#1b6576}progress{width:100%;accent-color:#28798a}a{color:#1c7086}
img{width:100%;display:block}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e2e9ec;padding:9px}th{color:#4b727d}ul{padding-left:22px}li{margin:7px 0}.scroll{overflow:auto}
@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:0 15px}section{padding:20px}h1{font-size:29px}}
</style><main><div class="eyebrow">Photovoltaic electroluminescence · evidence registry</div>
<h1>DINOv3 transfer under limited labels<br>and acquisition shift</h1>
<p class="lead">A controlled comparison of frozen features, LoRA, full fine-tuning and fully trained CNNs. All numbers below refer to recorded local work; pending experiments remain pending.</p>
<span class="badge">Research in progress · submission readiness not established</span>
<section><h2>Experiments running locally</h2><div class="grid">
<div class="card"><b>Stage 1 · Source-only calibration</b><div class="number" id="calibCount">Loading…</div><progress id="calibBar" max="36" value="0"></progress><div class="small" id="calibDetail"></div></div>
<div class="card"><b>Stage 2 · Real SCDD development</b><div class="number" id="studyCount">Loading…</div><progress id="studyBar" max="180" value="0"></progress><div class="small" id="studyDetail"></div></div>
<div class="card"><b>Classical feature controls</b><div class="number" id="classicCount">Loading…</div><progress id="classicBar" max="70" value="0"></progress><div class="small" id="classicDetail"></div></div>
<div class="card"><b>Matched pixel-supervision study</b><div class="number" id="denseCount">Loading…</div><progress id="denseBar" max="168" value="0"></progress><div class="small" id="denseDetail"></div></div>
<div class="card"><b>UCF-only reference transfer controls</b><div class="number" id="ucfrefCount">Loading…</div><progress id="ucfrefBar" max="9" value="0"></progress><div class="small" id="ucfrefDetail"></div></div>
<div class="card"><b>CNN source fitting stability</b><div class="number" id="cnnCount">Loading…</div><progress id="cnnBar" max="18" value="0"></progress><div class="small" id="cnnDetail"></div></div>
<div class="card"><b>Frozen pooling, resolution and scale</b><div class="number" id="reprCount">Loading…</div><progress id="reprBar" max="379" value="0"></progress><div class="small" id="reprDetail"></div></div>
<div class="card"><b>Local and frozen post-LoRA controls</b><div class="number" id="mechCount">Loading…</div><progress id="mechBar" max="125" value="0"></progress><div class="small" id="mechDetail"></div></div>
<div class="card"><b>Full published-method comparison</b><div class="number" id="pubCount">Loading…</div><progress id="pubBar" max="69" value="0"></progress><div class="small" id="pubDetail"></div></div>
<div class="card"><b>Complete inference efficiency</b><div class="number" id="effCount">Loading…</div><progress id="effBar" max="97" value="0"></progress><div class="small" id="effDetail"></div></div>
<div class="card"><b>ELPV supplementary review-label CV</b><div class="number" id="elpvCount">Loading…</div><progress id="elpvBar" max="24" value="0"></progress><div class="small" id="elpvDetail"></div></div>
</div><p class="small">Two learning-rate candidates per family; three paired seeds. Full-image D4 augmentation, batch 8, up to 2,000 optimizer updates, source-validation early stopping. Stage 2 starts after calibration is complete. Display refreshes every 30 seconds; no new model fitting occurs in this page.</p></section>
<section><h2>Real data and honest annotation accounting</h2><img src="figures/data_and_annotation_design.png" alt="Data counts by source and annotation budgets including validation">
<p class="small"><a href="figures/data_and_annotation_design.pdf">Vector PDF</a> · <a href="figures/data_and_annotation_design_source.json">Figure data</a> · <a href="scdd_data_audit.json">SCDD audit</a> · <a href="ucf_confirmation_audit.json">UCF audit</a></p>
<p>For each source-held-out experiment, training and validation use only the other two sources. All source-held-out comparisons use the 471 development images. The original 117 SCDD test images stay sealed.</p>
<p class="small">SCDD contains laboratory/collection-source groups; these are not verified physical module IDs. Its labels describe cracks anywhere in the annotated field of view. The class-stratified subsets simulate label availability; they do not measure prospective human annotation savings. UCF calibration labels are additional to the SCDD counts in the figure.</p>
<p class="small">The 507-image UCF supplemental confirmation cohort remains unused for model evaluation. It includes 27 items from the older still-sealed UCF test pool, so those two pools cannot count as independent replications. Real provenance is inferred from the authors’ train/test policy and remains a stated limitation. <a href="ucf_additional_leakage_audit.json">Additional D4 and cohort-overlap audit</a>.</p></section>
<section><h2>Independent samples and unseen domains are different claims</h2>
<p>UCF development labels calibrate the learning rates and classical regularization used in the SCDD experiments. Therefore, SCDD-trained models tested on new UCF acquisition groups provide group-disjoint confirmation in a dataset already exposed during parameter selection. They do not establish transfer to a never-observed dataset.</p>
<p>We preserve the predeclared UCF-only source-calibrated image models, and queue nine UCF-only normal-reference/position/aggregation fits after the current dense study. Their frozen-weight transfer to SCDD and PVEL forms a separate analysis direction. All predeclared families remain reportable; confirmation scores cannot choose the winner.</p>
<p class="small"><a href="model_selection_domain_exposure_audit.json">Selection-exposure audit</a> · <a href="pre_confirmation_transfer_scope_v1.json">Prospective transfer roles</a> · <a href="ucf_reference_transfer_v1/protocol.json">UCF-only reference controls</a></p></section>
<section><h2>Six transfer systems, identical label supervision</h2><div class="scroll"><table><thead><tr><th>System</th><th>Trainable feature extractor</th><th>Classification</th></tr></thead><tbody>
<tr><td>DINOv3-S frozen</td><td>None</td><td>CLS vector → linear layer</td></tr>
<tr><td>DINOv3-S + LoRA</td><td>Rank-8 Q/V in last 4 blocks</td><td>CLS vector → linear layer</td></tr>
<tr><td>DINOv3-S full</td><td>Entire feature extractor</td><td>CLS vector → linear layer</td></tr>
<tr><td>ResNet-18 / ResNet-50</td><td>Entire feature extractor</td><td>Pooled vector → linear layer</td></tr>
<tr><td>EfficientNet-B0</td><td>Entire feature extractor</td><td>Pooled vector → linear layer</td></tr>
</tbody></table></div><p class="small">All arms use image labels only and per-image LayerNorm. CNN pretrained BN statistics are retained. Different pretraining corpora mean these are comparisons of transfer systems, not a causal test of Transformer versus CNN architecture. Earlier pixel-supervised dense experiments remain a separate comparison.</p></section>
<section><h2>What the earlier 27-run head study established</h2><p>Adding hidden layers did not consistently improve the reused UCF pilot validation score. This supports keeping a simple linear head in the next controlled study, while testing how much feature adaptation and label availability matter.</p>
<p><a href="../advisor_head_study_20261016/report.html">Open the completed head/adaptation study</a></p><p class="small">Three seeds; checkpoints reloaded with identical saved predictions. These development results do not establish cross-dataset superiority or beat a published paper under a matched protocol.</p></section>
<section><h2>What still has to earn a paper</h2><ul>
<li>Finish strong CNN and DINOv3 comparisons across label budgets and acquisition sources.</li>
<li>Identify reproducible benefits or limitations, with acquisition-family uncertainty and failure cases.</li>
<li>Complete the registered matched-pixel-supervision study and normal-reference ablations; preserve negative results.</li>
<li>Freeze a final confirmation registry, evaluate the sealed cohorts, and report all outcomes.</li>
<li>Complete the primary-paper comparison, English manuscript and reproducibility package.</li></ul>
<p><b>No confirmation scores have been produced by this pipeline.</b> No acceptance, novelty or universal superiority is promised.</p></section>
<section><h2>Completed source calibration</h2><p>CALIBRATION_TEXT</p>CALIBRATION_FIGURE
<p class="small">Selected learning rates are frozen for the SCDD study. Validation was reused for this selection, so the plotted scores are development evidence. No claim of superiority over another paper follows from them.</p></section>
<section><h2>Annotation geometry is part of domain shift</h2>
<img src="figures/training_annotation_geometry.png" alt="Training-only annotation extent and component shape by dataset">
<p class="small">UCF annotations contain much larger filled regions than the thin structures in these SCDD training sources. Source populations, field of view and annotation conventions all differ; this plot does not identify their separate causes or measure physical crack severity.</p>
<details><summary>View real source-training images and ground-truth masks</summary><img src="figures/real_source_training_examples.png" alt="Training-only EL and annotated crack examples"></details></section>
<section><h2>New PVEL-AD audit changed the evaluation scope</h2>
<p>The author-posted archive was downloaded and all 15,853 annotated/author-normal candidate images passed decoding, byte-size and archive CRC checks. Exact rotation/reflection checking identified 639 duplicate clusters containing 1,452 identities. All members were quarantined before any model scoring.</p>
<p>Twenty-nine duplicate clusters differ in whether their XML includes a crack. Inspected examples show the same EL image with additional annotations in another file. Absence of a crack box is therefore insufficient evidence of a reliable hard negative in this archive.</p>
<div class="grid"><div class="card"><b>Authorized scoring pool before near-copy policy</b><p><span class="number">977</span> explicitly annotated crack positives<br><span class="number">11,349</span> author-reported normal images</p><div class="small">This unmodified pool is scored once. The primary analysis below excludes34 reviewed near-copy images. Measure source-threshold crack sensitivity and author-normal false alarms separately; this does not establish specificity against other defects.</div></div>
<div class="card"><b>Excluded from reliable negative claims</b><p><span class="number">2,075</span> other-defect images with uncertain crack-negative completeness</p><div class="small">The archive also contains 19,150 official-test images without XML; those are not assigned negative labels. Physical acquisition identifiers are unavailable.</div></div></div>
<p class="small"><a href="pvelad_metadata_audit.json">Archive/metadata audit</a> · <a href="pvelad_image_audit.json">Integrity/D4 audit</a> · <a href="pvelad_label_policy_v2.json">Restricted label policy</a> · <a href="figures/pvel_annotation_conflict_examples.png">Annotation-coverage examples</a>. No PVEL model scores have been produced.</p></section>
<section><h2>Novelty has to be narrower than a new backbone</h2>
<p>Recent primary work already studies DINOv3 transfer in industrial inspection and cross-domain EL diagnosis. Our contribution cannot be simply using DINOv3 or evaluating another dataset. The current tests examine the interaction of label budget, adaptation, acquisition shift and source-selected decisions.</p>
<p class="small"><a href="https://arxiv.org/abs/2605.23472">Gharbage et al., CVPRW 2026</a>: primary full text reviewed; transfer depends on adaptation and modality. <a href="https://doi.org/10.1016/j.enconman.2026.121958">Lu et al., 2026</a>: publisher abstract/introduction reviewed; detailed full-text comparison pending. The latter has a future issue date relative to this audit, while its publisher preview is already online. <a href="literature_update.json">Traceable screening and evidence limits</a>.</p></section>
<section><h2>Traceable records</h2><p><a href="calibration_v1/protocol.json">Calibration protocol</a> · <a href="calibration_v1/calibration_results.csv">Source-only partial results</a> · <a href="scdd_development_v1/design.json">Registered real-domain design</a> · <a href="evidence_status.json">Evidence status</a></p>
<p class="small">Primary data descriptions: <a href="https://github.com/TheMakiran/BenchmarkELimages">SCDD author repository</a> · <a href="https://doi.org/10.1016/j.sasc.2023.200048">Pratt et al., 2023</a> · <a href="https://github.com/ucf-photovoltaics/UCF-EL-Defect">UCF author repository</a> · <a href="https://doi.org/10.1109/JPHOTOV.2021.3131059">Fioresi et al., 2022</a> · <a href="https://github.com/binyisu/PVEL-AD">PVEL-AD author repository</a></p>
<p class="small"><a href="scope_coverage.json">Full-objective completion ledger</a> · <a href="pv_s3_reproduction_audit.json">PV-S3 paper/code comparison</a> · <a href="statistics_plan_v2.json">Current statistics guard</a> · <a href="localization_and_inspection_plan_v2.json">Current localization and label-policy plan</a></p></section>
<p class="small">Snapshot generated: SNAPSHOT · Local worker states refresh without computing confirmation predictions.</p></main>
<script>
async function refresh(){for(const [kind,path,total] of [['calib','calibration_v1',36],['study','scdd_development_v1',180]]){
try{const r=await fetch(path+'/status.json?time='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();
const n=s.completed_runs||0;document.getElementById(kind+'Count').textContent=n+' / '+total+' complete';document.getElementById(kind+'Bar').value=n;
const cur=s.current?Object.values(s.current).join(' · '):'';document.getElementById(kind+'Detail').textContent=s.state.replaceAll('_',' ')+' · '+cur+' · '+(s.updated_at||'');
}catch(e){document.getElementById(kind+'Detail').textContent='State unavailable; see the recorded protocol and logs.';}}
for(const [kind,path] of [['classic','classical_controls_v1'],['dense','dense_development_v1']]){
try{const r=await fetch(path+'/status.json?time='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();
const queued=s.state==='queued_after_image_study';const calibrating=kind==='classic'&&s.state==='calibrating';
const n=calibrating?(s.source_calibration_complete||0):(s.completed_runs||0);const total=calibrating?70:(kind==='classic'?180:168);
document.getElementById(kind+'Count').textContent=queued?'Queued · 168 planned':n+' / '+total+' complete';
document.getElementById(kind+'Bar').max=total;document.getElementById(kind+'Bar').value=n;
const cur=s.current?Object.values(s.current).join(' · '):queued?'Image study '+s.image_runs_complete+' / '+s.image_runs_planned:'';
document.getElementById(kind+'Detail').textContent=s.state.replaceAll('_',' ')+' · '+cur+' · '+(s.updated_at||s.at||'');
}catch(e){document.getElementById(kind+'Detail').textContent='State unavailable; see protocol and logs.';}}
for(const [kind,path,total] of [['ucfref','ucf_reference_transfer_v1',9],['cnn','cnn_stability_v1',18],['repr','representation_scale_v1',379],['mech','mechanistic_controls_v1',125],['pub','published_methods_v1',69],['eff','inference_efficiency_v1',97],['elpv','elpv_review_supplement_v1',24]]){
try{const r=await fetch(path+'/status.json?time='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();
const n=s.completed_jobs??s.completed_runs??0;document.getElementById(kind+'Count').textContent=s.state.startsWith('queued')?'Queued · '+total+' planned':n+' / '+total+' complete';
document.getElementById(kind+'Bar').value=n;document.getElementById(kind+'Detail').textContent=s.state.replaceAll('_',' ')+' · '+(s.at||'');
}catch(e){document.getElementById(kind+'Detail').textContent='See protocol and recorded worker state.';}}
}refresh();setInterval(refresh,30000);
</script></html>'''.replace('SNAPSHOT',html.escape(evidence['updated_at']))
    page=page.replace('CALIBRATION_TEXT','All 36 source-only runs are complete.' if calibration_done else 'Calibration is still running; no selected summary is available.')
    page=page.replace('CALIBRATION_FIGURE','<img src="figures/source_calibration_summary.png" alt="Source-only selected calibration AUROC with three seed points"><p class="small"><a href="figures/source_calibration_summary.pdf">Vector PDF</a> · <a href="figures/source_calibration_summary_source.json">Figure data</a></p>' if calibration_done else '')
    section='''<section><h2>First complete real-domain label-budget comparisons</h2>
<img src="figures/scdd_source_budget_progress.png" alt="Completed three-seed source-validation comparisons; missing budgets marked pending">
<p>At 5% training labels (18 images), LoRA mean source-validation AUROC is 0.809 versus 0.750 for frozen DINOv3. At 10% (36 images), the means are 0.889 and 0.801. Each point uses the same 120 validation images in addition to the training labels.</p>
<p class="small">All four internal budgets and six held-source image cells are complete. These are exploratory development scores. Independent confirmation and stronger converged feature controls are still required. Error bars show sample SD, not a confidence interval. Figure is a generated snapshot; live run counts are above.</p>
<p class="small"><a href="figures/scdd_source_budget_progress.pdf">Vector PDF</a> · <a href="figures/scdd_source_budget_progress_source.json">All seed values and run hashes</a></p></section>'''
    if development:
        page=page.replace('<section><h2>Real data and honest annotation accounting</h2>',section+'<section><h2>Real data and honest annotation accounting</h2>')
        page=page.replace('<section><h2>Independent samples and unseen domains are different claims</h2>', '''<section><h2>Does the model distinguish cracks within each acquisition source?</h2>
<img src="figures/source_shortcut_diagnostic.png" alt="Pooled versus equal-source within-source AUROC with a metadata-only source-prior diagnostic">
<p>At 20% training labels, LoRA has pooled AUROC 0.924 and within-source macro AUROC 0.912; EfficientNet-B0 has 0.927 and 0.901. The slight ranking change shows why we retain both summaries and the held-source experiments.</p>
<p class="small">The source-prior diagnostic uses only the selected training labels and the acquisition-source identity; it is not an image classifier. A pooled/within-source gap reflects several possible effects and does not prove causal shortcut use. These reused validation results are exploratory, with three seeds/subsets and sample SD. The CNN shown was preselected using UCF calibration.</p>
<p class="small"><a href="figures/source_shortcut_diagnostic.pdf">Vector PDF</a> · <a href="source_shortcut_diagnostic.json">All six systems, source priors and per-source results</a></p></section>
<section><h2>Independent samples and unseen domains are different claims</h2>''')
    if held['complete_cells']:
        page=page.replace('<section><h2>Does the model distinguish cracks within each acquisition source?</h2>', '''<section><h2>First completed source-held-out comparison exposes a reliability gap</h2>
<img src="figures/first_held_source_development.png" alt="All six systems on the first complete acquisition-source holdout; AUROC, target recall and review fractions">
<p>Training on the other two acquisition sources, LoRA obtains mean target AUROC 0.761 on 151 ARTS development images. Its source-calibrated high-recall threshold gives target recall 0.615 and review fraction 0.320. Full DINOv3 has AUROC 0.784, recall 0.756 and review fraction 0.501. None of these numbers establishes reliable high-recall deployment.</p>
<p class="small">20% of the permitted source training pool; three paired seeds. These are different fitted source-held-out models, not the same model as the internal source-validation curve. All six systems and all complete holdout cells are included. This is exploratory development among the 471 images; the original 117 test images remain sealed.</p>
<p class="small"><a href="figures/first_held_source_development.pdf">Vector PDF</a> · <a href="held_source_development_complete_cells.json">All seeds, frozen-threshold metrics and provenance</a></p></section>
<section><h2>Acquisition shift has different effects across sources</h2>
<img src="figures/held_source_development_matrix.png" alt="All complete acquisition-held-out development cells, three seed means and sample deviations; missing cells marked pending">
<p>With 20% source-training labels, LoRA reaches mean AUROC 0.964 on held-out CFVS and 0.761 on ARTS. The ARTS comparison against EfficientNet-B0 has a mean AUROC difference of 0.055, with a conditional acquisition-family 95% interval spanning roughly −0.048 to 0.141. A numerical advantage is not consistent evidence of superiority in every source.</p>
<p class="small">Exploratory development; intervals are conditional on the three fitted models, with seed variability reported separately. The paired acquisition-family analysis follows the existing statistics plan; these are not verified physical modules or adjusted confirmatory tests. <a href="held_source_paired_uncertainty.json">All registered contrasts and uncertainty</a> · <a href="figures/held_source_development_matrix.pdf">Vector PDF</a>.</p></section>
<section><h2>Higher point estimates do not always establish a reliable gain</h2>
<img src="figures/held_source_paired_contrasts.png" alt="All four predeclared paired LoRA contrasts in each analyzed complete development cell, with conditional acquisition-family95 percent intervals">
<p>With full source-training labels on CFVS, LoRA has mean AUROC0.992 and EfficientNet-B0 has0.980, but the paired conditional interval for their difference spans zero. The ARTS20% comparison also spans zero. Each completed cell retains all four baseline contrasts; no baseline is omitted because it is competitive.</p>
<p class="small">The intervals reflect capture-family uncertainty conditional on the fitted models; they do not establish universal or causal superiority. <a href="figures/held_source_paired_contrasts.pdf">Vector PDF</a> · <a href="figures/held_source_paired_contrasts_source.json">Underlying differences and intervals</a>.</p></section>
<section><h2>Does the model distinguish cracks within each acquisition source?</h2>''')
    page=page.replace('<section><h2>What the earlier 27-run head study established</h2>', '''<section><h2>Controls registered before final confirmation</h2>
<p><b>Converged feature classifiers:</b> HOG + 42 statistics and frozen DINOv3 CLS vectors each feed Logistic Regression, Linear SVM and RBF SVM. Training folds, nested label budgets and D4 views match the image-only neural study. Classical source grids are wider; tuning budgets are not identical. Repeated full-budget deterministic fits do not represent independent optimizer seeds.</p>
<p><b>Complete localization systems:</b> adapted DeepLabv3+/ResNet50 and a four-depth DINOv3 decoder use the same image/pixel supervision. Frozen, LoRA, full adaptation and normal-reference variants test the contribution of references, positional matching and top-local evidence. All DINO decoder heads start identically. The study includes 24 UCF calibration fits and 144 real-domain development fits, queued after image-model checkpoint verification.</p>
<p class="small">Validation pixel masks are charged separately from image labels. Matched SCDD localization remains in progress; the completed UCF source calibration is reported separately. The adapted supervised DeepLabv3+ does not reproduce the PV-S3 dual-teacher semi-supervised method.</p>
<p><b>Pooling, resolution and model scale:</b> 13 frozen representations compare CLS, geometric-area-weighted patch means and concatenation; DINOv3-S uses 224/336/448 inputs and DINOv3-B uses 336. The same three-C Logistic Regression grid applies to all representations. Frozen ImageNet ResNet18 also gets Linear/RBF SVM controls. The registered study has 49 UCF calibration trials and 330 unique SCDD development fits. Full-budget deterministic probes are saved once.</p>
<p class="small">Official Small/Base and ResNet18 passed CPU encoding checks on one observed UCF training image, including exclusion of register tokens. The Small CLS differs from its existing MPS cache by at most 9.1×10⁻⁶. This is an implementation check, not a Base classification result. <a href="representation_scale_v1/protocol.json">Representation/scale protocol</a> · <a href="representation_scale_v1/cpu_encoder_smoke.json">Encoder smoke check</a>.</p>
<p><b>Explicit local mechanisms:</b> 36 patch-MLP fits, 23 normal-distance memory controls and 66 fresh-head post-LoRA fits/calibrations are now registered. In the post-LoRA stage, only the original budgeted source images adapt the encoder; its Q/V updates are merged, frozen and re-encoded before building references. Reference/no-reference decoder heads start identically. All target confirmation scores remain sealed.</p>
<p class="small">The patch baseline max-preserves thin crack labels at patch level; its objective differs from full-resolution segmentation. The normal-distance baseline uses mask-informed references and a source noncrack-FPR cutoff, not a crack posterior or annotation-free learning. Both differences remain explicit. <a href="mechanistic_controls_v1/protocol.json">125-run mechanism protocol</a> · <a href="mechanism_cpu_smoke_v1/summary.json">Source-only pipeline checks, not performance results</a>.</p>
<p><b>Full published-method control:</b> the MIT-licensed PS-MT deep-stem ResNet50/DeepLabv3+ architecture is retained, with 41.76 million parameters per network. The semi-supervised arm trains a student using two EMA teachers, teacher-decoder VAT, confidence-weighted CE, strong photometry and CutMix. All floating BN state and integer counters are saved losslessly and verified with fresh-model predictions.</p>
<p class="small">The registered 69-fit protocol covers source calibration, a fixed released-VAT numerical sensitivity, supervised/SSL acquisition-held-out development, and matched UCF20% DINO controls. Primary inference uses teacher1 as in released PV-S3; teacher-logit ensemble is a correlated additional view at the same source-selected checkpoint. Canonical3-class labels, whole-field336 inputs, batch2/MPS, source selection and update schedules are explicit adaptations. This is not a reproduction of the paper score. Actual336px CPU training/reload checks passed, with no claim of predictive accuracy. <a href="published_methods_v1/protocol.json">Full-method protocol</a> · <a href="published_implementation_smoke_336_v1/summary.json">Implementation-only checks</a> · <a href="published_architecture_source.json">Source/license audit</a>.</p>
<p><b>Complete inference cost:</b> 97 predeclared model/seed/view cases will each process the same 351 SCDD and 430 UCF source-training images, at batch sizes 1 and 8. Fifteen additional 20%-label reference/distance cases test how annotation-dependent normal memory changes cost. Three full passes without instrumentation provide primary latency; a separate component pass checks encoding, matching, classification, native-map reconstruction and memory.</p>
<p class="small">Actual source CPU engineering checks passed for frozen DINOv3 and the complete teacher network. Formal timing remains queued until all own training/probe workers finish. MPS sampled allocations are observed maxima, not guaranteed GPU peaks or CUDA VRAM; RSS includes model loading. Teacher1 inference discards the inactive teacher after full CPU archive validation, with exact prediction equality checked. <a href="inference_efficiency_v1/protocol.json">97-case source-only efficiency protocol</a> · <a href="inference_efficiency_implementation_v1/summary.json">Engineering evidence, not formal timing results</a>. Independent confirmation now also requires the complete verified benchmark.</p>
<p class="small"><a href="classical_controls_v1/protocol.json">Classical protocol</a> · <a href="dense_development_v1/protocol.json">Dense protocol</a></p></section>
<section><h2>What the earlier 27-run head study established</h2>''')
    page=page.replace('<section><h2>Traceable records</h2>', '''<section><h2>English development manuscript</h2>
<p>The current v04 manuscript contains audited data/label scope, complete published-method and efficiency specifications, MANUSCRIPT_CELL_COUNT complete acquisition-held-out cells and MANUSCRIPT_CONTRAST_COUNT neural conditional paired contrasts, plus the complete classical comparisons and dense source calibration. Its standalone source compiled successfully with the native LaTeX editor. Independent confirmation, complete SCDD localization, formal published-method training and full inference measurements remain pending.</p>
<p class="small"><a href="../../manuscript/research_manuscript_v04.tex">Editable standalone LaTeX manuscript</a> · <a href="../../manuscript/manuscript_v04_inputs.json">Archived table inputs and compiler status</a>. This is a development draft, not a submission-ready manuscript.</p></section>
<section><h2>Traceable records</h2>''')
    manuscript=read_json(ROOT/'manuscript/manuscript_v04_inputs.json')
    page=page.replace('MANUSCRIPT_CELL_COUNT',str(manuscript['completed_cells'])).replace('MANUSCRIPT_CONTRAST_COUNT',str(manuscript.get('paired_contrasts',16)))
    if (BASE/'image_failure_analysis_v1/report.html').exists():
        page=page.replace('<section><h2>English development manuscript</h2>', '''<section><h2>What is missed, and what triggers a false alarm?</h2>
<img src="image_failure_analysis_v1/development_failure_examples.png" alt="Selected native EL and ground-truth defect annotations; errors of all six models at the20 percent source-training budget">
<p>All45 ARTS crack-positive development images have annotated crack footprints below0.5% of known pixels. CFVS and CSIR include larger masks. Acquisition, field of view and annotation footprint therefore co-vary; these data cannot isolate the cause of the source performance difference.</p>
<p>The examples include a CFVS negative flagged by all LoRA seeds and none of the EfficientNet seeds. This is a selected counterexample, not a representative sample. Every system/seed/subgroup count remains available.</p>
<p class="small">Ground-truth overlays are annotations, not model attention or predicted heatmaps. This retrospective analysis opens only471 observed development masks; no sealed confirmation images. <a href="image_failure_analysis_v1/report.html">Read the complete error, subgroup and probability report</a> · <a href="image_failure_analysis_v1/development_failure_examples.pdf">Vector PDF</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    page=page.replace('<section><h2>Higher point estimates do not always establish a reliable gain</h2>', '''<section><h2>Inspection means detecting cracks and controlling review workload</h2>
<img src="figures/held_source_inspection_tradeoff.png" alt="All six systems and three seeds: target crack recall versus target review fraction across three acquisition sources and two label budgets">
<p>Points near the upper left combine higher recall with fewer reviews. LoRA often improves on frozen features, but source-frozen thresholds behave differently by acquisition channel. A high AUROC alone cannot establish a useful review workload or meet a target-recall requirement.</p>
<p class="small">All36 family/cell means and108 seed points are retained. The dashed95% line is the source criterion, not a guaranteed target rate. <a href="figures/held_source_inspection_tradeoff.pdf">Vector PDF</a> · <a href="figures/held_source_inspection_tradeoff_source.json">All underlying scores and source bindings</a>.</p></section>
<section><h2>Higher point estimates do not always establish a reliable gain</h2>''')
    page=page.replace('<section><h2>English development manuscript</h2>', '''<section><h2>Final-confirmation execution and analysis are prepared</h2>
<p>The source-model inventory is an unreviewed preview. No final registry or confirmation observation has been created. The workflow requires all source controls, verified full inference costs and the separate ELPV supplement to finish before source-reviewed choices can be frozen.</p>
<p>After every frozen model/cohort job completes, an output audit checks saved IDs, logits, source decisions and calibration. Analysis reports every declared contrast, including unsupported endpoints as not estimable; it cannot select a replacement endpoint. Native positive-image Dice and normal-structure FPR use image/capture-family summaries. Pooled pixel AP stays a whole-cohort point; PVEL receives no invented acquisition-group interval.</p>
<p class="small"><a href="confirmation_preparation_v1/candidate_catalog.json">Source-only candidate inventory</a> · <a href="confirmation_preparation_v1/actual_guard_check.json">Actual unreviewed-plan refusal</a>. Fixture checks do not establish model accuracy.</p>
<p><b>ELPV supplement:</b> the original1968 course-development images and four outer folds are preserved. HOG/statistics and DINOv3 CLS each feed Logistic, Linear SVM and RBF SVM with hyperparameters and scaling nested inside each training fold. This means24 reported outer fits,480 inner-grid fits, and zero reads of the656 course test images. It is a review-label task, not crack truth or independent confirmation. It runs after the cost benchmark to avoid contaminating timings.</p>
<p class="small"><a href="elpv_review_supplement_v1/protocol.json">Supplementary CV protocol</a></p></section>
<section><h2>English development manuscript</h2>''')
    page=page.replace('<section><h2>Novelty has to be narrower than a new backbone</h2>', '''<section><h2>Near-copy review changes the prospective primary population</h2>
<img src="figures/preinference_near_copy_evidence.png" alt="Native PVEL pairs, differing crack boxes and fixed-scale differences; no model outputs">
<p>A nonlearned audit across15,963 images found238 screened similarities and18 stronger candidates. Native review identifies17 PVEL near-copy pairs with differing boxes. Their34 members are excluded from the primary view before any confirmation model scoring. The unchanged12,326-image pool remains for sensitivity analysis.</p>
<div class="scroll"><table><thead><tr><th>Predeclared view</th><th>Total</th><th>Known crack</th><th>Author normal</th></tr></thead><tbody>
<tr><td>Primary, reviewed near-copy members excluded</td><td>12,292</td><td>943</td><td>11,349</td></tr>
<tr><td>Original authorized pool, sensitivity</td><td>12,326</td><td>977</td><td>11,349</td></tr>
<tr><td>All screen members excluded, sensitivity</td><td>11,927</td><td>933</td><td>10,994</td></tr></tbody></table></div>
<p>One scoring pass supplies all views; sensitivities are not independent replications. Candidate-confirmation pixels have been inspected for integrity, while model scores remain unopened. Origin, physical-module identity, every cropped overlap and pretraining membership are not established by this screen.</p>
<img src="figures/capture_family_sensitivity.png" alt="Eight comparisons under original and possible-related capture-family resampling">
<p>One highly similar pair in SCDD training motivates a possible-related-family merge for retrospective uncertainty analysis. All eight point differences remain unchanged. The LoRA versus EfficientNet interval at20% labels still spans zero. Three weaker source/confirmation pairs with different printed module IDs are retained.</p>
<p class="small"><a href="whole_frame_similarity_audit_v1/report.html">Complete integrity review and native sheets</a> · <a href="confirmation_preparation_v1/cohort_analysis_policy_v1.json">Prospective population policy</a> · <a href="confirmation_preparation_v1/population_policy_verification.json">Fixed-example eligibility verification</a> · <a href="figures/preinference_near_copy_evidence.pdf">Vector audit figure</a> · <a href="figures/capture_family_sensitivity.pdf">Vector sensitivity figure</a></p></section>
<section><h2>Novelty has to be narrower than a new backbone</h2>''')
    if (BASE/'classical_development_analysis_v1/results.json').exists():
        page=page.replace('<section><h2>Real data and honest annotation accounting</h2>', '''<section><h2>Converged SVM and Logistic controls are now complete</h2>
<img src="figures/classical_and_lora_source_holdouts.png" alt="Every classical family and LoRA at both held-source annotation budgets">
<p>All70 source calibration trials and180 development fits have completed. Reloading the6 selected UCF models plus180 SCDD models reproduces saved source logits exactly; all108 held-source arrays also agree. Source annotation identities match the neural comparison.</p>
<p>At100% source-training labels, DINOv3 CLS + Linear SVM mean held-source AUROC is0.806 on ARTS,0.956 on CFVS and0.821 on CSIR. LoRA means are0.853,0.992 and0.928. The ARTS LoRA-minus-SVM conditional interval spans zero; the comparison is exploratory development evidence.</p>
<p class="small">All6 classical families,60 cell summaries and36 paired contrasts are retained. Full-budget classical seed records are deterministic duplicates, not3 independent optimizer repetitions. Search grids and training objectives differ. <a href="classical_development_analysis_v1/report.html">Complete classical comparison</a> · <a href="figures/classical_and_lora_source_holdouts.pdf">Vector figure</a> · <a href="classical_development_analysis_v1/results.json">Reload, seed and interval evidence</a>.</p></section>
<section><h2>Real data and honest annotation accounting</h2>''')
    if dense_uncertainty_path.exists():
        entries=read_json(dense_uncertainty_path)['complete_cells']
        ten=next((r for r in entries if r['fold']=='internal' and r['budget']==.1),None)
        if ten:
            result=read_json(ROOT/ten['path']);render=read_json((ROOT/ten['path']).parent/'latest_figure.json')
            image_path=str((ROOT/render['figure_directory']/'paired_differences.png').relative_to(BASE))
            paired=next(r for r in result['comparisons'] if r['first']=='deeplabv3plus_full' and r['second']=='dinov3_dense_lora')
            auc=paired['endpoints']['AUROC']['estimate'];dice=paired['endpoints']['positive_image_Dice']['estimate']
            ai=auc['acquisition_family_percentile_ci95'];di=dice['acquisition_family_percentile_ci95']
            page=page.replace('<section><h2>English development manuscript</h2>',f'''<section><h2>Classification gains can coexist with weaker localization</h2>
<img src="{image_path}" alt="All seven paired10-percent dense model comparisons, all seeds and conditional capture-family intervals">
<p>At10% training masks, LoRA minus CNN image AUROC is{auc['mean_pair_difference']:+.3f} (conditional interval[{ai[0]:+.3f}, {ai[1]:+.3f}]), while positive-image Dice is{dice['mean_pair_difference']:+.3f} ([{di[0]:+.3f}, {di[1]:+.3f}]). The same three fitted seed pairs and120 reused source-validation images support both endpoints. This does not establish independent superiority.</p>
<p>Normal references improve frozen-decoder source Dice at5% and10%, but do not consistently outperform stronger CNN/LoRA controls. At5%, increased Dice also coexists with more normal-structure false positives. We retain all declared comparisons and endpoints, including adverse results. The two budgets reuse the same evaluation images; they are not independent domain replications.</p>
<p class="small">Intervals resample74 provided capture families, conditional on fitted seeds. They do not include a population of training runs or prove physical-module independence. Normal-structure FPR is displayed in percentage points; JSON/CSV retain raw ratios. Pooled native AP is point-only, without pixel resampling. Training masks are additional to120 validation masks and upstream UCF labels. <a href="dense_development_uncertainty_v1/report.html">Every complete-cell interval, seed and endpoint</a> · <a href="dense_reliability_interpretation_v1.json">Evidence-bound interpretation</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    curve_latest=BASE/'dense_budget_curve_v1/latest.json'
    if curve_latest.exists():
        curve_index=read_json(curve_latest);curve_path=ROOT/curve_index['source']
        curve=read_json(curve_path)
        if sha256(curve_path)!=curve_index['sha256'] or curve['identity']['generator_sha256']!=sha256(ROOT/'build_dense_budget_curve.py'):
            raise RuntimeError('Dense budget-curve source/renderer changed')
        for binding in curve['identity']['cell_bindings']+curve['figures']:
            if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Dense budget-curve binding changed')
        curve_image=str((ROOT/curve['figures'][0]['path']).relative_to(BASE))
        page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>All four internal mask budgets: classification and localization diverge</h2>
<img src="{curve_image}" alt="All five fully crossed dense systems at four mask budgets, every fitted seed and sample SD">
<p>At full training masks, LoRA image AUROC is 0.961 versus CNN 0.849, but native pixel AP is 0.224 versus 0.249 and positive-image Dice is 0.149 versus 0.198. Better whole-image ranking therefore does not guarantee better crack localization. These remain reused source-validation results.</p>
<p>The curves include all 60 fits from five fully crossed systems and all three seeds at 18, 36, 70 and 351 training masks. Another 12 component fits at 20% and 100% are retained in their registered comparisons. Error bars show sample SD, not confidence intervals. Every point additionally uses 120 validation masks and 574 upstream UCF calibration annotations.</p>
<p>At 100%, normal reference minus frozen Dice is +0.027, but reference minus CNN is −0.059. Top-local evidence minus mean evidence Dice is −0.032 in the fitted-pair mean, with all three seed differences negative, while its normal-structure FPR is 0.0895 percentage points lower. The tradeoff differs from the 20% mean-evidence failure; no pooling or position rule is retuned mid-study.</p>
<p class="small">All four budgets reuse the same 120 images and 74 provided capture families. Conditional comparisons do not establish population-over-training-seed stability, independent superiority or a physical cause. <a href="{curve_image.replace('.png','.pdf')}">Vector learning curves</a> · <a href="dense_development_uncertainty_v1/report.html">All four completed source cells and every declared contrast</a> · <a href="dense_real_source_gallery_v1/035f4ca2fef23a9703da/shared_native_rois.png">Full-budget native examples</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    transfer_index=BASE/'dense_transfer_readout_v1/index.json'
    if transfer_index.exists():
        transfer=read_json(transfer_index)
        if transfer['builder_sha256']!=sha256(ROOT/'build_dense_transfer_readout.py'):
            raise RuntimeError('Transfer readout builder changed')
        first_transfer=next((x for x in transfer['complete_held_source_cells'] if x['fold']=='holdout_ARTS' and x['budget']==.2),None)
        if first_transfer:
            transfer_path=ROOT/first_transfer['path'];r=read_json(transfer_path)
            if sha256(transfer_path)!=first_transfer['sha256']:raise RuntimeError('Transfer readout changed')
            for binding in r['identity']['bindings']+r['figures']:
                if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Transfer readout source or figure changed')
            transfer_image=str((ROOT/r['figures'][0]['path']).relative_to(BASE))
            all_dense_results=[read_json(ROOT/e['path']) for e in read_json(dense_uncertainty_path)['complete_cells']]
            dense_contrasts=sum(len(v['comparisons']) for v in all_dense_results)
            dense_endpoints=sum(len(c['endpoints']) for v in all_dense_results for c in v['comparisons'])
            page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>First dense held-source result: ranking improves, localization remains unresolved</h2>
<img src="{transfer_image}" alt="Every source-selected checkpoint and all three fitted seeds on source validation versus held ARTS; four systems and three endpoints">
<p>All four systems and three seeds are complete on151 ARTS development images, including45 crack positives and84 provided capture families. They use51 training masks and65 source-validation masks from other sources, plus upstream UCF calibration exposure. Mean LoRA image AUROC is0.825 versus CNN0.569. The LoRA-minus-CNN paired AUROC difference is+0.256, with conditional interval[+0.170,+0.328]; all three fitted differences are positive.</p>
<p>Positive-image Dice is0.089 for LoRA versus CNN0.103: the paired difference is−0.014 with interval[−0.038,+0.006]. Two seed pairs favor CNN and one favors LoRA. The mean recall advantage also has only one favorable seed pair. Normal reference improves a frozen decoder, but its normal-structure FPR exceeds LoRA in all three fitted pairs. None of this establishes universal reliability or an original-paper score reproduction.</p>
<p>CNN seed22 has source-validation native AP0.274 and Dice0.249, but ARTS AP0.0014 and Dice0. At the fixed0.5 cutoff, all45 ARTS positives have empty predicted masks, while6,021 pixels are predicted as crack on other images. This concrete failure comes from original native counts. Source/ARTS crack-pixel prevalence differs (0.284% versus0.033%), and source validation is not training-set fit; the result alone cannot identify optimizer underfitting, a physical cause or an isolated causal domain effect.</p>
<p class="small">Same fitted models, two observed populations; faint lines retain every seed. Conditional intervals resample capture families, conditional on fitted models. Original independent confirmation remains unopened. <a href="{transfer_image.replace('.png','.pdf')}">Vector source-to-held figure</a> · <a href="{str(transfer_path.parent.relative_to(BASE))}/all_fitted_seed_outcomes.csv">Every fitted-seed source/target outcome</a> · <a href="dense_development_uncertainty_v1/report.html">All{dense_contrasts} completed dense contrasts and{dense_endpoints} endpoint summaries</a> · <a href="dense_transfer_readout_v1/index.json">All complete held-source readouts</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    if transfer_index.exists():
        full_transfer=next((x for x in transfer['complete_held_source_cells'] if x['fold']=='holdout_ARTS' and x['budget']==1.),None)
        if full_transfer:
            full_path=ROOT/full_transfer['path'];full_readout=read_json(full_path)
            if sha256(full_path)!=full_transfer['sha256']:raise RuntimeError('Full-source readout changed')
            for binding in full_readout['identity']['bindings']+full_readout['figures']:
                if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Full-source readout binding changed')
            full_image=str((ROOT/full_readout['figures'][0]['path']).relative_to(BASE))
            labels={'deeplabv3plus_full':'Adapted CNN','dinov3_dense_frozen':'DINO frozen','dinov3_dense_lora':'DINO LoRA','dinov3_dense_reference':'Normal reference'}
            operating_rows=[]
            for c in dense_cells['complete_cells']:
                if c['fold']!='holdout_ARTS':continue
                for model in c['models']:
                    rates=[]
                    for fit in model['fits']:
                        evaluation=(ROOT/fit['run']).parent/'target_development.json'
                        binding=next(x for x in c['identity']['bindings'] if x['run']==fit['run'])
                        if sha256(evaluation)!=binding['evaluation_sha256']:raise RuntimeError('Operating decision input changed')
                        m=read_json(evaluation)['metrics'];negative=m['fp']+m['tn']
                        if negative<=0:raise RuntimeError('Do not substitute undefined image FPR')
                        rates.append([m['recall'],m['fp']/negative,m['review_rate']])
                    means=np.mean(rates,axis=0)
                    operating_rows.append('<tr><td>'+str(int(100*c['budget']))+'%</td><td>'+labels[model['family']]+'</td>'+''.join(f'<td>{100*v:.1f}%</td>' for v in means)+'</tr>')
            page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>Full-source ARTS follow-up: high recall still flags most images</h2>
<img src="{full_image}" alt="All four systems and all three fitted seeds at255 source training masks, same checkpoints on source validation and held ARTS">
<p>Both ARTS training-mask budgets are now complete. At255 source-training masks, LoRA mean AUROC is0.882 versus CNN0.584; every fitted seed favors LoRA. The paired difference is+0.297 (conditional interval[+0.193,+0.394]). Positive-image Dice is0.093 versus CNN0.133, with paired difference−0.040 [−0.059,−0.022]; two seeds favor CNN and one slightly favors LoRA. LoRA normal-structure pixel FPR is0.466 percentage points lower than CNN. These endpoints express different benefits and costs.</p>
<div class="scroll"><table><thead><tr><th>Source training masks</th><th>System</th><th>Image recall</th><th>Crack-negative image FPR</th><th>Images flagged</th></tr></thead><tbody>{''.join(operating_rows)}</tbody></table></div>
<p>LoRA flags80.1% of all ARTS images while retrieving97.0% of crack-positive images. CNN attains the same mean recall but flags91.6%. LoRA reduces the flagged fraction relative to CNN, yet ranking performance alone does not establish a small inspection burden. Crack-negative image FPR is distinct from normal-structure pixel FPR. Negative images may contain other defects, and these fractions do not measure unnecessary human work or economic cost.</p>
<p class="small">Same151 ARTS images/45 positives/84 provided capture families across budgets; not independent replications or a causal annotation-cost experiment. Fractions mean51 versus255 training masks plus65 source-validation masks and574 upstream UCF calibration annotations. Original models, pixel cutoff and source-selected image thresholds remain unchanged; no ARTS threshold fitting. <a href="{full_image.replace('.png','.pdf')}">Vector same-checkpoint comparison</a> · <a href="dense_development_uncertainty_v1/report.html">Every conditional contrast and endpoint</a> · <a href="dense_transfer_readout_v1/index.json">All completed source/held readouts</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    if transfer_index.exists():
        # Extend to every later complete source cell, including adverse outcomes.
        for entry in transfer['complete_held_source_cells']:
            if entry['fold']=='holdout_ARTS':continue
            path=ROOT/entry['path'];readout=read_json(path)
            if sha256(path)!=entry['sha256']:raise RuntimeError('Held-source readout changed')
            for binding in readout['identity']['bindings']+readout['figures']:
                if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Held-source figure or source changed')
            cell=next(c for c in dense_cells['complete_cells'] if c['cell_id']==readout['cell_id'])
            conditional_entry=next(e for e in read_json(dense_uncertainty_path)['complete_cells'] if e['cell_id']==cell['cell_id'])
            conditional_path=ROOT/conditional_entry['path'];conditional=read_json(conditional_path)
            if sha256(conditional_path)!=conditional_entry['sha256']:raise RuntimeError('Conditional held-source result changed')
            rendered=read_json(conditional_path.parent/'latest_figure.json')
            conditional_image=str((ROOT/rendered['figure_directory']/'paired_differences.png').relative_to(BASE))
            transfer_image=str((ROOT/readout['figures'][0]['path']).relative_to(BASE))
            label=cell['fold'].removeprefix('holdout_');percentage=100*cell['budget']
            labels={'deeplabv3plus_full':'Adapted CNN','dinov3_dense_frozen':'DINO frozen','dinov3_dense_lora':'DINO LoRA','dinov3_dense_reference':'Normal reference'}
            rows=[]
            for model in cell['models']:
                rates=[]
                for fit in model['fits']:
                    evaluated=(ROOT/fit['run']).parent/'target_development.json'
                    binding=next(b for b in cell['identity']['bindings'] if b['run']==fit['run'])
                    if sha256(evaluated)!=binding['evaluation_sha256']:raise RuntimeError('Held-source operating decision changed')
                    m=read_json(evaluated)['metrics'];denominator=m['fp']+m['tn']
                    if denominator<=0:raise RuntimeError('Undefined image FPR must not become zero')
                    rates.append([m['recall'],m['fp']/denominator,m['review_rate']])
                values=[model['mean'][k] for k in ['image_AUROC','native_pixel_AP','positive_image_Dice']]
                rows.append('<tr><td>'+labels[model['family']]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+''.join(f'<td>{100*v:.1f}%</td>' for v in np.mean(rates,axis=0))+'</tr>')
            pair=next(c for c in conditional['comparisons'] if c['first']=='deeplabv3plus_full' and c['second']=='dinov3_dense_lora')
            reference=next(c for c in conditional['comparisons'] if c['first']=='dinov3_dense_frozen' and c['second']=='dinov3_dense_reference')
            def paired_text(comparison,endpoint):
                estimate=comparison['endpoints'][endpoint]['estimate'];interval=estimate['acquisition_family_percentile_ci95']
                return f"{estimate['mean_pair_difference']:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]"
            failed=[]
            for target in readout['all_seed_population_records']:
                if target['population']!='target' or target['predicted_crack_pixels']!=0:continue
                source=next(r for r in readout['all_seed_population_records'] if r['population']=='source' and r['family']==target['family'] and r['seed']==target['seed'])
                if source['predicted_crack_pixels']==0:
                    failed.append(f"{labels[target['family']]} seed {target['seed']} already has an entirely empty predicted mask on all {readout['source_support']['images']} source-validation images, and also on all {readout['target_support']['images']} held images, at the fixed 0.5 pixel cutoff. Selected round {source['selected_round']} of {source['completed_rounds']}; original predicted-pixel counts establish this outcome.")
            source=readout['source_support'];target=readout['target_support']
            training_masks=readout['all_seed_population_records'][0]['training_masks']
            failure_text=('<p>'+html.escape(' '.join(failed))+' This is not established as a target-only failure. All fitted arms remain in the comparison; the source optimization/stability controls must also be considered.</p>') if failed else ''
            page=page.replace('<section><h2>English development manuscript</h2>', f'''<section id="dense-held-{label}-{percentage:g}"><h2>Held {label} · {percentage:g}%: ranking, pixels and operating decisions</h2>
<img src="{transfer_image}" alt="All four matched systems and all three fitted seeds, same checkpoints on source validation and held {label}">
<div class="scroll"><table><thead><tr><th>System</th><th>Image AUROC</th><th>Native pixel AP</th><th>Positive Dice</th><th>Image recall</th><th>Crack-negative image FPR</th><th>Images flagged</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>LoRA minus CNN: AUROC {paired_text(pair,'AUROC')}; Dice {paired_text(pair,'positive_image_Dice')}. Reference minus frozen: AUROC {paired_text(reference,'AUROC')}; Dice {paired_text(reference,'positive_image_Dice')}. Brackets are conditional 95% capture-family intervals, not uncertainty over future training runs. Pooled pixel AP remains point-only.</p>
{failure_text}
<img src="{conditional_image}" alt="Every declared complete {label} dense contrast, all seed differences and conditional capture-family intervals">
<p class="small">{target['images']} held images / {target['positive_images']} positives / {conditional['provided_capture_families']} provided capture families. {training_masks} source training masks + {source['images']} source-validation masks + 574 upstream UCF calibration annotations. These are observed development sources, not the unopened independent confirmation. Image thresholds remain source-selected; image FPR and normal-structure pixel FPR are distinct. Flagged fraction is not measured human cost. <a href="{transfer_image.replace('.png','.pdf')}">Vector same-checkpoint figure</a> · <a href="{conditional_image.replace('.png','.pdf')}">Vector conditional comparisons</a> · <a href="{str(path.relative_to(BASE))}">Every fitted seed, population and input hash</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    annotation_support_latest=BASE/'native_annotation_support_v1/latest.json'
    if annotation_support_latest.exists():
        support_index=read_json(annotation_support_latest);support=read_json(ROOT/support_index['results'])
        support_figure=str((ROOT/support['figures'][0]['path']).relative_to(BASE))
        scdd_support=support['cohorts']['SCDD'];ucf_support=support['cohorts']['UCF']
        support_copies=support['same_native_annotations_verified_in_run_summaries']
        page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>Pixel AP needs annotation-prevalence and case-weighting context</h2>
<img src="{support_figure}" alt="All SCDD provided source supports and cumulative annotated crack pixels over each complete positive validation cohort">
<p>SCDD has {100*scdd_support['constant_pixel_score_AP']:.3f}% annotated crack pixels among known pixels; UCF has {100*ucf_support['constant_pixel_score_AP']:.3f}%. These fractions are the AP of a constant pixel score, rather than trained-model results. Different source fits, label conventions and prevalences prevent ranking the datasets using their AP numbers alone.</p>
<p>CFVS provides 59.3% of SCDD crack-positive images but 86.7% of its annotated crack pixels. The six largest positive masks provide {100*scdd_support['top10pct_positive_image_crack_pixel_share']:.1f}% of crack pixels. This motivates retaining equally weighted positive-image Dice and small-case analyses alongside pooled AP. It does not decompose AP or identify a physical cause of a model difference.</p>
<p class="small">The same native GT counts and identities are verified across {support_copies['UCF']} UCF and {support_copies['SCDD']} SCDD saved source evaluations: 264 unique annotated images, not {sum(support_copies.values())} independent annotation replications. No new inference or final-test access. <a href="native_annotation_support_v1/report.html">Both complete cohorts, every image count and vector figure</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    component_index=BASE/'dense_mechanism_interpretation_v1/index.json'
    if component_index.exists():
        components=read_json(component_index)
        first=next((e for e in components['complete_component_cells'] if e['budget']==.2),None)
        if first:
            component=read_json(ROOT/first['path'])
            def selected_component(name,endpoint):
                return next(r for r in component['component_contrasts'] if r['first']==name and r['endpoint']==endpoint)
            position=selected_component('dinov3_reference_no_position','positive_image_Dice')
            pooling=selected_component('dinov3_reference_mean_evidence','positive_image_Dice')
            source=next(b for b in component['input_bindings'] if '/dense_development_uncertainty_v1/' in b['path'])
            component_render=read_json((ROOT/source['path']).parent/'latest_figure.json')
            component_figure=ROOT/component_render['figure_directory']/'paired_differences.png'
            if not component_figure.exists():raise RuntimeError('Require exact current component-cell uncertainty figure')
            paired_image=str(component_figure.relative_to(BASE))
            page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>20% component ablations: average gains can hide seed failures</h2>
<img src="{paired_image}" alt="All nine prespecified20-percent dense contrasts, three fitted-seed differences and conditional capture-family intervals">
<p>The full comparison now includes all seven systems and three fitted seeds, using 70 training masks plus 120 reused validation masks and upstream UCF calibration exposure. Position-minus-no-position mean Dice is {position['mean_difference']:+.3f}; one paired seed improves and two worsen. Pooled native AP moves differently. This does not establish a reliable benefit from the position penalty.</p>
<p>Top 5%-minus-mean evidence mean Dice is {pooling['mean_difference']:+.3f}, but only one seed favors top-local evidence. The selected mean-evidence seed 33 checkpoint predicts no crack pixels above 0.5 on all 120 native validation images, including 54 positives; its pooled native AP is still 0.144. The other two Dice differences favor mean evidence. This is evidence about the selected models and cutoff, not a proven physical mechanism or universal pooling advantage.</p>
<p class="small">Intervals condition on these three fitted models and 74 provided capture families; they do not resample future training outcomes. Original protocols, thresholds and fixed gallery cases are unchanged. <a href="dense_development_uncertainty_v1/report.html">All contrasts and endpoints</a> · <a href="dense_real_source_gallery_v1/report.html">All fixed native cases</a> · <a href="dense_mechanism_interpretation_v1/index.json">Seed differences and original native predicted-pixel counts</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    effort_latest=BASE/'source_training_effort_v1/latest.json'
    if effort_latest.exists():
        effort_index=read_json(effort_latest);effort=read_json(ROOT/effort_index['results'])
        effort_figure=str((ROOT/effort['figures'][0]['path']).relative_to(BASE))
        page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>Training fewer parameters is different from a faster complete system</h2>
<img src="{effort_figure}" alt="All72 internal image fits: fitted versus total parameters, completed optimizer updates and training presentations per original image">
<p>Image LoRA fits 49,537 parameters versus 21,596,929 for full DINOv3 image adaptation. The pretrained encoder remains in the inference model. All six systems, four budgets and three fitted seeds are retained; patience updates after checkpoint selection count as actual work.</p>
<p>The source audit verifies {effort['source_fits']} completed image/dense run histories and annotation counts. Frozen training uses cached D4 vectors; cache construction is excluded from its solver timer. Counts and diagnostic seconds do not establish end-to-end training or inference speed, FLOPs or memory gains. The 97-case complete inference benchmark remains pending.</p>
<p class="small"><a href="source_training_effort_v1/report.html">Complete current source-work audit, all counts and vector figure</a>. Only fully complete dense cells enter this snapshot; future published/other-stage training costs require their own completed records.</p></section>
<section><h2>English development manuscript</h2>''')
    if (BASE/'method_provenance_v1/review.json').exists():
        page=page.replace('<section><h2>English development manuscript</h2>', '''<section><h2>Existing methods versus the research contribution</h2>
<p>Original method attribution now covers ten works, with selected full-text sections read for five. PatchCore and AnomalyDINO precede normal patch memory; SuperADD already uses DINOv3 with nominal feature banks and acquisition variation. A newer backbone or a normal-memory combination is therefore not claimed as new.</p>
<p>Our supervised, mask-informed crack task differs from nominal-only anomaly detection. The registered comparisons test annotation requirements, localization failures, source-frozen decisions and transfer. They do not reproduce the three anomaly papers exactly or rank their MVTec/VisA scores against EL scores.</p>
<p class="small"><a href="method_provenance_v1/report.html">Original methods, read sections and applicability limits</a>. Metadata-only and abstract-only sources remain explicitly marked; SuperADD retains workshop-challenge technical-report status. No additional experiment or confirmation access.</p></section>
<section><h2>English development manuscript</h2>''')
    access_review=BASE/'source_access_and_reproduction_v1/review.json'
    if access_review.exists():
        page=page.replace('<section><h2>English development manuscript</h2>', '''<section><h2>Data, model and third-party source availability</h2>
<p>Versioned author-source evidence now distinguishes repository-code terms, image/data access and DINOv3 model terms. The official and pinned model-license bytes match. The future code-only profile retains required notices and author acquisition recipes; the existing private implementation capsule remains distinct from a public distribution.</p>
<p>A four-file setup recipe reacquires exact reviewed PS-MT architecture/notice and PV-S3 numerical-oracle bytes in a separate destination. Reuse and no-overwrite probes pass. No source is executed, no test is skipped, and no raw EL data, model weights or final scores are acquired by this check.</p>
<p class="small"><a href="source_access_and_reproduction_v1/report.html">Primary sources, versions and concrete reconstruction evidence</a>. This is an availability/reproduction preparation result, not model-performance evidence or an executed public release.</p></section>
<section><h2>English development manuscript</h2>''')
    fresh_path=BASE/'reproducibility_preparation_v1/latest_fresh_environment.json'
    if fresh_path.exists() and read_json(fresh_path).get('success'):
        fresh=read_json(fresh_path)
        page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>Fresh pinned environment passes copied-source checks</h2>
<p>A separate newly installed environment from official PyPI passes all{fresh['passed']} copied-source implementation checks. All{fresh['fresh_pinned_packages']} exact pinned dependencies come from the new environment; dependency consistency also passes. Python-level guards deny original site-packages, original research/course data, pretrained/auth caches and network access during tests.</p>
<p class="small">The first installation succeeded but its test launcher omitted the copied project path and denied its own proof output. That failed attempt is preserved; an amended launcher passed in a second independently installed environment. Same machine/base Python; this is not OS isolation, cross-device determinism or reproduction of research performance. <a href="reproducibility_preparation_v1/latest_fresh_environment.json">Actual fresh-environment evidence</a> · <a href="reproducibility_preparation_v1/dependency_availability_v1/latest.json">Official53-release availability audit</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    gallery_path=BASE/'dense_real_source_gallery_v1/index.json'
    if gallery_path.exists() and read_json(gallery_path)['pages']:
        gallery=read_json(gallery_path)['pages'][0]
        page=page.replace('<section><h2>English development manuscript</h2>', f'''<section><h2>Real EL localization: success and missed thin cracks</h2>
<img src="dense_real_source_gallery_v1/{gallery['folder']}/shared_native_rois.png" alt="Same real EL fields, manual crack annotations and all completed matched models with shared fixed color scales">
<p>The first complete5% source cell retains all five systems and all eight originally fixed cases. Main panels use the first negative and smallest/largest annotated footprint among those positives; model scores never choose cases or crops. The thin ARTS crack receives little response, whereas the large CFVS crack has stronger response. This illustrates source-specific failures; three selected cases cannot estimate a population effect.</p>
<p class="small">Fixed0–1 softmax scale; cyan manual truth, white dashed score>0.5, gray ignored regions. No per-image normalization. Seed11 float16 maps are illustrative; numerical metrics use original full-float evaluation. <a href="dense_real_source_gallery_v1/report.html">All eight cases and original case metrics</a> · <a href="dense_real_source_gallery_v1/{gallery['folder']}/shared_native_rois.pdf">Vector PDF</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    capsule_path=BASE/'reproducibility_preparation_v1/latest_capsule.json'
    if capsule_path.exists() and read_json(capsule_path).get('offline_test_status')=='passed':
        page=page.replace('<section><h2>English development manuscript</h2>', '''<section><h2>Copied-source offline checks pass</h2>
<p>An independent local source snapshot passes allCAPSULE_TEST_COUNT implementation checks with model-hub access disabled. It includes real dataset/protocol metadata and upstream numerical-oracle code, alongside synthetic training fixtures. Raw EL images/masks, learned/pretrained weights and original feature caches are absent.</p>
<p>The latest check also enables Python audit-hook guards against original project/course files, pretrained/auth caches and network access. It passes with zero blocked-access attempts. This is a Python-level dependency check, not an operating-system sandbox.</p>
<p class="small">The first snapshot missed a required parent-protocol metadata file and failed one check; it is preserved. A new snapshot includes the protocol without skipping the check. These tests reuse the original Python environment; they do not demonstrate fresh-machine installation, bitwise cross-device training replication or complete paper reproduction. <a href="reproducibility_preparation_v1/latest_capsule.json">Snapshot and actual offline-test evidence</a>.</p></section>
<section><h2>English development manuscript</h2>''')
        page=page.replace('CAPSULE_TEST_COUNT',str(read_json(capsule_path)['offline_test_passed']))
    if (BASE/'dense_source_calibration_analysis_v1/summary.json').exists():
        page=page.replace('<section><h2>Annotation geometry is part of domain shift</h2>', '''<section><h2>Dense source calibration completed: endpoints tell different stories</h2>
<p>All24 UCF source fits completed before selecting the four architectures’ learning rates. The selected full-adaptation DINOv3 mean native pixel AP is0.736, versus0.684 for the adapted DeepLabv3+ baseline. LoRA mean pixel AP is0.685, while its mean positive-image Dice0.372 is lower than the CNN’s0.422. This does not establish an independent or universal advantage.</p>
<img src="dense_source_calibration_analysis_v1/selected_source_native_crops.png" alt="Fixed native source cases and shared annotation-defined crops, with the same0–1 crack-score scale for every model">
<p>Same430 training masks and144 reused validation masks; means describe three selected source runs. The illustrated seed11 cases come from the originally fixed gallery. Small-crack misses remain visible, and all8 examples are retained. No model score selects the cases, crop coordinates or display range.</p>
<p class="small">The source validation set selects both rate and checkpoint; this table and these maps are development evidence. The adapted supervised DeepLabv3+ differs from the queued complete semi-supervised published-method control. <a href="dense_source_calibration_analysis_v1/report.html">All seed values, all8 grid candidates and full native gallery</a> · <a href="dense_source_calibration_analysis_v1/selected_source_native_crops.pdf">Vector PDF</a>.</p></section>
<section><h2>Annotation geometry is part of domain shift</h2>''')
    if (BASE/'architecture_and_exposure_v1/source_and_probe.json').exists():
        page=page.replace('<section><h2>Six transfer systems, identical label supervision</h2>', '''<section><h2>What the joint model actually computes</h2>
<img src="architecture_and_exposure_v1/implemented_dino_joint_architecture.png" alt="Actual four-depth DINOv3 decoder, optional normal residual input, image head and local evidence fusion">
<p>The normal-reference and reference-free arms retain the same1,387,269 decoder/head parameters. Their771-value local slots contain query features, residuals, coordinates and distance. A64-channel projection joins512 projected multidepth channels. The image head combines768 global features with one local-evidence value.</p>
<p class="small">A synthetic CPU shape/gradient probe verified these dimensions against the actual stored official configuration and the completed source decoder count. It loads no pretrained weights or real images and provides no performance evidence. Equal parameter counts do not prove that nearest-reference matching itself causes a gain; a matching-specific claim would need a suitable correspondence control. <a href="architecture_and_exposure_v1/implemented_dino_joint_architecture.pdf">Vector architecture PDF</a> · <a href="architecture_and_exposure_v1/source_and_probe.json">Actual code bindings and probe</a>.</p>
<details><summary>Model-specific fitting versus study-wide development exposure</summary><img src="architecture_and_exposure_v1/source_selection_and_confirmation_exposure.png" alt="Source fitting, model selection and the three frozen confirmation populations"><p>Both UCF and SCDD development outcomes inform this study. UCF-only fitted weights provide transfer to SCDD, while SCDD is already observed during method development. PVEL model outcomes remain unseen before freeze, despite raw image-integrity review.</p></details></section>
<section><h2>Six transfer systems, identical label supervision</h2>''')
    if (BASE/'dense_source_localization_diagnostic_v1/results.json').exists():
        page=page.replace('<section><h2>Annotation geometry is part of domain shift</h2>', '''<section><h2>High image detection can coexist with zero correct crack overlap</h2>
<img src="dense_source_localization_diagnostic_v1/image_detection_localization_disconnect.png" alt="Source image recall and image detection with zero native crack overlap for every selected architecture and footprint group">
<p>The7 smallest-footprint UCF source-validation positives expose a failure: full DINO adaptation has mean zero-overlap fraction0.905 at pixel cutoff0.5, despite pooled native pixel AP0.736 over the complete validation cohort. LoRA’s mean fraction is0.952; the adapted CNN’s is0.619. These are seven images under three fitted seeds, not21 independent samples.</p>
<p>All systems’ mean source image recall0.966 partly reflects source-threshold selection on the same validation set. Zero overlap can also mean a misplaced prediction. It does not prove a physical failure mechanism or target-domain performance.</p>
<p class="small">The future confirmation policy now fixes these area boundaries and secondary counts before model scoring. Empty strata remain null; primary endpoints/comparisons stay unchanged. Normal-distance responses retain their source-calibrated cutoff meaning. <a href="dense_source_localization_diagnostic_v1/report.html">All58 positive-image outcomes, all60 subgroup records and diagnostic figures</a> · <a href="confirmation_preparation_v1/secondary_native_failure_policy_v1.json">Pre-scoring secondary policy</a>.</p></section>
<section><h2>Annotation geometry is part of domain shift</h2>''')
    if dense_cells and dense_cells['complete_cells']:
        cell=dense_cells['complete_cells'][0]
        path=str((ROOT/cell['cell_path']).parent.relative_to(BASE))
        section=f'''<section><h2>First complete real SCDD dense comparison</h2>
<img src="{path}/comparison.png" alt="All predeclared dense systems and all three fitted seeds at the first complete real-data budget">
<p>Every predeclared system and all three fitted seeds are complete in this cell. It uses5% of training masks (18 images), plus120 source-validation masks and the additional UCF calibration exposure. The source-validation set also selects checkpoints.</p>
<p>Normal reference mean Dice is0.127 versus0.053 for the matched frozen decoder, while native pixel AP is0.161 versus0.157. Endpoints disagree in the size of the numerical improvement. This does not establish a stable mechanism, independent advantage or deployment-ready segmentation.</p>
<p class="small">Currently{len(dense_cells['complete_cells'])} of10 source cells are complete. Partial cells are omitted from finished comparisons. <a href="dense_complete_cells_v1/report.html">Every completed real-data cell</a> · <a href="dense_development_complete_cells.json">All seed values and immutable cell hashes</a>.</p></section>'''
        page=page.replace('<section><h2>Real data and honest annotation accounting</h2>',section+'<section><h2>Real data and honest annotation accounting</h2>')
    native_gallery_index=BASE/'dense_held_native_gallery_v1/index.json'
    if native_gallery_index.exists():
        native=read_json(native_gallery_index)
        if native['complete_cells']!=6 or native['native_maps']!=192:raise RuntimeError('All6 native source galleries required')
        for item in native['pages']:
            data=item['figure_inputs']
            if not data['research_held_source_figure']:raise RuntimeError('Cannot present layout-check pixels as held-source results')
            for asset in data['figures']:
                if sha256(ROOT/asset['path'])!=asset['sha256']:raise RuntimeError('Native source figure changed')
        preview=next(p for p in native['pages'] if p['fold']=='holdout_CFVS' and p['budget']==1.)
        image_path=str((ROOT/preview['figure_inputs']['figures'][0]['path']).relative_to(BASE))
        case_latest=BASE/'dense_held_native_gallery_v1/original_case_metrics/latest.json'
        case_link=''
        if case_latest.exists():
            ci=read_json(case_latest);cp=ROOT/ci['path']
            if sha256(cp)!=ci['sha256']:raise RuntimeError('Fixed original case links changed')
            cases=read_json(cp);csv_path=str((ROOT/cases['csv']['path']).relative_to(BASE))
            case_link=f' · <a href="{csv_path}">Original counts and image decisions for all192 displayed maps</a>'
        section=f'''<section><h2>Verified native crack maps across all three sources</h2>
<img src="{image_path}" alt="Held CFVS full-budget example: native EL, shared ground truth ROI and four source-trained models on identical fixed cases">
<p>All24 whole-cohort prediction replays agree exactly with the original logits, native counts and evaluated floating-point metrics. The192 fixed-case model maps depict24 distinct images across budgets and systems. This preview shows CFVS at full source training; every source and both budgets are retained.</p>
<p>Cyan is ground truth. White is the displayed prediction contour above0.5. Heatmaps share a fixed0–1 scale; the eight-case supplements show raw probability maps. Float16 figures illustrate seed11, while original full-float predictions supply the three-seed results.</p>
<p><a href="dense_held_native_gallery_v1/report.html">All six comparisons and all fixed cases</a>{case_link}.</p></section>'''
        page=page.replace('<section><h2>Real data and honest annotation accounting</h2>',section+'<section><h2>Real data and honest annotation accounting</h2>')
    ucf_reference_index=BASE/'ucf_reference_readout_v1/latest.json'
    if ucf_reference_index.exists():
        ref_index=read_json(ucf_reference_index);ref_path=ROOT/ref_index['path']
        if sha256(ref_path)!=ref_index['sha256']:raise RuntimeError('UCF reference snapshot changed')
        ref=read_json(ref_path)
        if ref['identity']['builder_sha256']!=sha256(ROOT/'build_ucf_reference_readout.py'):
            raise RuntimeError('UCF reference report builder changed')
        for binding in ref['identity']['bindings']+ref['figures']:
            if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('UCF reference source changed')
        image_path=str((ROOT/ref['figures'][0]['path']).relative_to(BASE))
        section=f'''<section><h2>All UCF source-reference controls complete</h2>
<img src="{image_path}" alt="All21 original selected UCF source fits: four parent systems and three reference variants, all three fitted seeds">
<p>All9 registered reference fits and all12 selected parent fits are shown. Each uses430 training masks and the same144 validation masks. The reference variants retain position-aware top-five-percent evidence, remove the position penalty, or replace local top evidence with a mean.</p>
<p class="small">Validation labels select checkpoints and parent hyperparameters: these are optimistic source summaries, not independent confirmation. UCF lacks the annotated normal-structure support needed for a structure-specific FPR. No additional inference is performed by this readout. <a href="{image_path[:-4]}.pdf">Vector PDF</a> · <a href="{str(ref_path.parent.relative_to(BASE))}/all21_source_runs.csv">All21 original source records</a>.</p></section>'''
        page=page.replace('<section><h2>Real data and honest annotation accounting</h2>',section+'<section><h2>Real data and honest annotation accounting</h2>')
    transfer_summary_index=BASE/'dense_transfer_summary_v1/latest.json'
    if transfer_summary_index.exists():
        summary_index=read_json(transfer_summary_index);summary_path=ROOT/summary_index['path']
        if sha256(summary_path)!=summary_index['sha256']:raise RuntimeError('Dense transfer summary changed')
        summary=read_json(summary_path)
        if summary['identity']['builder_sha256']!=sha256(ROOT/'build_dense_transfer_summary.py'):
            raise RuntimeError('Dense summary renderer changed')
        for binding in summary['identity']['input_bindings']+summary['figures']:
            if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Dense summary evidence changed')
        image_path=str((ROOT/summary['figures'][0]['path']).relative_to(BASE))
        cells=summary['cells']
        if len(cells)!=6:raise RuntimeError('Only all-six-cell summaries are displayed')
        auc_wins=sum(c['endpoints']['AUROC']['estimate']['mean_pair_difference']>0 for c in cells)
        ap_losses=sum(c['endpoints']['pooled_native_pixel_AP']['estimate']['mean_pair_difference']<0 for c in cells)
        section=f'''<section><h2>All six dense transfer comparisons in one figure</h2>
<img src="{image_path}" alt="All six observed held-source cells: LoRA minus adapted CNN for AUROC, native pixel AP, positive-image Dice and normal-structure false positives">
<p>LoRA mean image AUROC is higher in {auc_wins}/6 cells, while adapted CNN mean native pixel AP is higher in {ap_losses}/6. Dice and normal-structure false positives depend on the source and training budget. Every fitted seed is retained; native AP has no bootstrap interval.</p>
<p class="small">These are three acquisition sources reused at two training-mask budgets, not six independent datasets. The CNN is our matched adapted DeepLabv3+ control; this is not a direct comparison with a published headline score. Stability, published-method and independent-confirmation studies remain separate. <a href="{image_path[:-4]}.pdf">Vector PDF</a> · <a href="{str(summary_path.parent.relative_to(BASE))}/all_paired_values.csv">All72 paired endpoint values</a>.</p></section>'''
        page=page.replace('<section><h2>Real data and honest annotation accounting</h2>',section+'<section><h2>Real data and honest annotation accounting</h2>')
    operating_index=BASE/'dense_operating_readout_v1/latest.json'
    if operating_index.exists():
        index=read_json(operating_index);result_path=ROOT/index['path']
        if sha256(result_path)!=index['sha256']:raise RuntimeError('Source operating snapshot changed')
        operating=read_json(result_path)
        for binding in operating['identity']['input_bindings']+operating['figures']:
            if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Source operating figure changed')
        image_path=str((ROOT/operating['figures'][0]['path']).relative_to(BASE))
        summary='Every shown point retains the original source-selected threshold; further sources remain pending until every system and seed completes.'
        points=operating['all_fitted_points'];full={}
        for source in ['ARTS','CFVS']:
            selected=[p for p in points if p['fold']=='holdout_'+source and p['budget']==1. and p['family']=='dinov3_dense_lora']
            if selected:
                if sorted(p['seed'] for p in selected)!=[11,22,33]:raise RuntimeError('No partial source decision summary')
                full[source]={k:float(np.mean([p[k] for p in selected])) for k in ['recall','negative_image_FPR']}
        if len(full)==2:
            a=full['ARTS'];c=full['CFVS']
            summary=f"At full source training, CFVS LoRA recall is{100*c['recall']:.1f}% with{100*c['negative_image_FPR']:.1f}% crack-negative image FPR; ARTS LoRA recall is{100*a['recall']:.1f}% with{100*a['negative_image_FPR']:.1f}% image FPR. These are separately source-fitted models on different observed cohorts. The contrast does not identify a physical or isolated causal acquisition mechanism."
        page=page.replace('<section><h2>English development manuscript</h2>',f'''<section><h2>Image decisions: high recall can carry very different false-positive rates</h2>
<img src="{image_path}" alt="Every complete dense held-source cell, four systems, all fitted seeds, recall and crack-negative image false-positive rate at original source thresholds">
<p>All {len(operating['all_fitted_points'])} fitted seed points from {operating['complete_held_source_cells']} complete source cells are retained. {summary}</p>
<p class="small">Faint markers show every seed; diamonds show their mean. Pending sources remain visible. Each point uses its own original source-selected threshold; these are not ROC curves or confidence intervals. Image FPR differs from normal-structure pixel FPR. Negative images may contain other defects, and flagged fractions do not establish human or economic costs. <a href="{image_path.replace('.png','.pdf')}">Vector figure</a> · <a href="{str((result_path.parent/'every_fitted_seed.csv').relative_to(BASE))}">Every original integer count and fitted operating point</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    source_control_index=BASE/'completed_source_controls_v1/latest.json'
    if source_control_index.exists():
        entry=read_json(source_control_index);result_path=ROOT/entry['path']
        if sha256(result_path)!=entry['sha256']:raise RuntimeError('Completed source-control readout changed')
        controls=read_json(result_path);qa_path=result_path.parent/'visual_verification.json'
        if not qa_path.exists() or not read_json(qa_path)['all3_actual_PNGs_inspected']:
            raise RuntimeError('Inspect all source-control figures before dashboard integration')
        for asset in controls['figures']:
            if sha256(ROOT/asset['path'])!=asset['sha256']:raise RuntimeError('Source-control figure changed')
        directory=str(result_path.parent.relative_to(BASE))
        cnn_mean=next(x['mean_AUROC'] for x in controls['CNN_summary'] if x['encoder']=='resnet50' and x['learning_rate']==1e-5)
        page=page.replace('<section><h2>English development manuscript</h2>',f'''<section><h2>Completed CNN optimization and frozen-feature controls</h2>
<p>All18 additional CNN learning-rate fits and all379 representation fits are complete. Retaining the original controls gives27 CNN fits plus3 image-only LoRA references,49 UCF representation calibrations and330 unique development probes. The readout checks409 existing checkpoint hashes and589 original saved prediction arrays; it performs no new model inference.</p>
<img src="{directory}/cnn_learning_rate_controls.png" alt="Every CNN learning-rate setting and fitted seed with original image-only LoRA source reference">
<p>At encoder LR1e-5, ResNet-50 mean source AUROC is {cnn_mean:.3f}, versus {controls['image_LoRA_mean_AUROC']:.3f} for the original image-only LoRA. The same120 validation images select checkpoints and motivated these controls. This concerns image-only optimization; it does not reverse the separate dense transfer comparison or establish independent superiority.</p>
<img src="{directory}/representation_held_sources.png" alt="Every frozen representation, classifier and held-source budget with subset and single-fit distinctions">
<p class="small">15 classifier families /13 frozen representations. Lower-budget entries average3 nested training subsets; full-budget deterministic fits are saved once. All source-held outcomes are observed development. Upstream430+144 UCF labels and additional SCDD validation labels are disclosed. <a href="{directory}/report.html">All results, internal matrix and exposure limits</a> · <a href="{directory}/all27_cnn_source_fits.csv">Every CNN fit</a> · <a href="{directory}/all330_representation_development_fits.csv">Every frozen-feature fit</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    mechanism_index=BASE/'mechanistic_complete_cells_v1/index.json'
    if mechanism_index.exists():
        mechanism=read_json(mechanism_index);cards=[]
        if mechanism['builder_sha256']!=sha256(ROOT/'build_mechanistic_progress_readout.py'):
            raise RuntimeError('Mechanism reporter changed')
        for entry in mechanism['complete_cells']:
            result_path=ROOT/entry['path'];result=read_json(result_path)
            if sha256(result_path)!=entry['sha256']:raise RuntimeError('Complete mechanism result changed')
            qa=read_json(result_path.parent/'visual_verification.json')
            if not qa['all2_actual_PNGs_inspected'] or qa['result']!=entry:
                raise RuntimeError('Inspect all complete mechanism plots before integration')
            for a in result['figures']:
                if sha256(ROOT/a['path'])!=a['sha256']:raise RuntimeError('Mechanism figure changed')
            directory=str(result_path.parent.relative_to(BASE))
            cards.append(f'<h3>{entry["fold"]} · {100*entry["budget"]:g}% source training masks</h3><img src="{directory}/all_completed_arms.png" alt="Every original dense and newly registered mechanism model, all fitted seeds"><img src="{directory}/all_postlora_paired_seeds.png" alt="All three matched post-LoRA reference minus fresh-decoder seed differences">')
        twenty_entry=next((e for e in mechanism['complete_cells'] if e['fold']=='internal' and e['budget']==.2),None)
        additional=''
        if twenty_entry:
            result=read_json(ROOT/twenty_entry['path']);means={r['family']:r['mean'] for r in result['all_model_summaries']}
            fresh=means['dinov3_postlora_no_reference'];ref=means['dinov3_postlora_reference'];parent=means['dinov3_dense_lora']
            additional=f'<p>At20% source masks, reference lowers Dice in all3paired seeds: mean{fresh["Dice"]:.3f} to{ref["Dice"]:.3f}, with nativeAP{fresh["AP"]:.3f}/{ref["AP"]:.3f}. The fresh reference-free decoder improves mean Dice{parent["Dice"]:.3f} to{fresh["Dice"]:.3f} and nativeAP{parent["AP"]:.3f} to{fresh["AP"]:.3f} versus its original LoRA parent, favorable in all3fitted pairs. This adds a second optimization stage and encoder freezing; it does not isolate layer depth or equal-compute advantage.</p>'
        full_entry=next((e for e in mechanism['complete_cells'] if e['fold']=='internal' and e['budget']==1.),None)
        if full_entry:
            result=read_json(ROOT/full_entry['path']);means={r['family']:r['mean'] for r in result['all_model_summaries']}
            fresh=means['dinov3_postlora_no_reference'];parent=means['dinov3_dense_lora']
            additional+=f'<p>The complete100% internal source cell contradicts extrapolation of that gain: fresh-head AP{parent["AP"]:.3f} to{fresh["AP"]:.3f}, Dice{parent["Dice"]:.3f} to{fresh["Dice"]:.3f}; AP/Dice/AUROC are adverse in all3parent pairs. Structure FPR also decreases, a tradeoff. The reference seed11 predicts zero crack pixels above0.5 across all120 validation images. Failures remain in all aggregates.</p>'
        arts_entry=next((e for e in mechanism['complete_cells'] if e['fold']=='holdout_ARTS' and e['budget']==1.),None)
        if arts_entry:
            result=read_json(ROOT/arts_entry['path']);means={r['family']:r['mean'] for r in result['all_model_summaries']}
            fresh=means['dinov3_postlora_no_reference'];parent=means['dinov3_dense_lora']
            additional+=f'<p>Held ARTS100%: fresh-head native AP{parent["AP"]:.3f} to{fresh["AP"]:.3f} improves in all3parent pairs, but Dice{parent["Dice"]:.3f} to{fresh["Dice"]:.3f} improves in only1pair and worsens in2. Mean Dice gain is driven by one pair. Two ARTS budgets reuse151 observed target-development images; the source-fitted models/populations also differ from the internal cell. This is not independent confirmation or an isolated acquisition-effect estimate.</p>'
        page=page.replace('<section><h2>English development manuscript</h2>',f'''<section><h2>Adding new heads after LoRA: complete mechanism controls</h2>
<p>{len(mechanism['complete_cells'])}/10 complete cells retain every original dense model and all four registered local/distance/fresh-head arms. Each post-LoRA pair freezes the same adapted encoder and trains identically initialized fresh decoder/image heads, with or without source-only reference inputs. This adds a second training stage.</p>
{''.join(cards)}
<p>At10% source masks, all three post-LoRA pairs increase normal-structure pixel FPR with reference. Mean Dice rises partly because the no-reference seed33 predicts no crack pixels at the fixed cutoff on all120 validation images. Its pixel AP remains nonzero; the paired reference arm has slightly lower AP. The weaker normal-distance image scores are retained in their original direction.</p>
{additional}
<p class="small">All3 fitted seed differences are shown, including failures. No CI, p-value or independent confirmation is asserted. Pooled pixel AP is point-only. Distance and learned-softmax cutoffs differ. These120 validation images are reused across budgets; reference matching-specific causal benefit is unestablished. <a href="mechanistic_complete_cells_v1/report.html">Every complete model, seed, endpoint and lineage check</a> · <a href="mechanistic_complete_cells_v1/index.json">All pending cells</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    summary_index=BASE/'complete_mechanism_summary_v1/latest.json'
    if summary_index.exists():
        e=read_json(summary_index);p=ROOT/e['path'];summary=read_json(p);qa=read_json(p.parent/'visual_verification.json')
        if sha256(p)!=e['sha256'] or qa['result_sha256']!=e['sha256'] or not qa['all_actual_PNGs_inspected']:
            raise RuntimeError('Require reviewed complete mechanism summary')
        for a in summary['figures']:
            if sha256(ROOT/a['path'])!=a['sha256']:raise RuntimeError('Complete mechanism summary figure changed')
        directory=str(p.parent.relative_to(BASE))
        page=page.replace('<section><h2>English development manuscript</h2>',f'''<section><h2>All125 mechanism fits complete: gains and failures together</h2>
<img src="{directory}/all10_paired_head_contrasts.png" alt="Both head contrasts, all10 development cells, every fitted seed and original adverse results">
<p>Normal reference raises native AP in every CFVS fitted pair at20% and100% masks. Full-budget CFVS Dice also rises in all3pairs, while the adapted CNN retains higher mean native AP/Dice. In CSIR,20% reference improves AP/Dice and structure FPR in all3pairs; at100%, mean Dice falls and structure FPR rises in all3pairs. Neither reference nor retraining a head gives a universal improvement.</p>
<p class="small">Ten design cells reuse four overlapping observed evaluation cohorts; they are not ten independent domains. Fitted-seed points and means, no CI/pvalue; native AP point-only. Extra head training adds a stage and does not isolate layer depth. All6 endpoints are retained. <a href="complete_mechanism_summary_v1/report.html">Complete summary and60pair CSV</a>. Published-method training, formal costs and independent confirmation remain required.</p></section>
<section><h2>English development manuscript</h2>''')
    native_diagnostic_index=BASE/'mechanistic_source_native_gallery_v1/latest.json'
    if native_diagnostic_index.exists():
        entry=read_json(native_diagnostic_index);result_path=ROOT/entry['path'];g=read_json(result_path)
        if sha256(result_path)!=entry['sha256'] or not read_json(result_path.parent/'visual_verification.json')['all4_actual_PNGs_inspected']:
            raise RuntimeError('Inspect all native mechanism diagnostic figures first')
        for a in g['figures']:
            if sha256(ROOT/a['path'])!=a['sha256']:raise RuntimeError('Native diagnostic figure changed')
        directory=str(result_path.parent.relative_to(BASE))
        page=page.replace('<section><h2>English development manuscript</h2>',f'''<section><h2>Original saved native maps: fresh heads can suppress crack predictions</h2>
<img src="{directory}/shared_native_ROIs.png" alt="Same original source images and GT-only column; CNN and LoRA parent versus new heads on a fixed linear score scale">
<p>Full source budget, original seed11 convention, unchanged first-negative/smallest/largest positive cases. Cyan GT is shown only in its own column. White model contours mean saved score above0.5; no cyan truth outline is repeated on prediction panels. This outcome-motivated source diagnostic uses original saved float16 maps, with quantitative metrics from original full-float evaluations.</p>
<p class="small">All8original cases and all11model families remain in3raw-map sheets (88original maps). Same0..1scale and0.65overlay opacity; no per-image normalization or new model inference. Every seed remains in quantitative comparisons. <a href="mechanistic_source_native_gallery_v1/report.html">All cases/families and full-float numeric links</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    illustrated_index=ROOT/'output/pdf/latest_development_pdf.json'
    if illustrated_index.exists():
        entry=read_json(illustrated_index);receipt_path=ROOT/entry['path']
        if sha256(receipt_path)!=entry['sha256']:raise RuntimeError('Illustrated PDF receipt changed')
        receipt=read_json(receipt_path);visual_path=receipt_path.parent/'visual_verification.json'
        if visual_path.exists():
            visual=read_json(visual_path);pdf=ROOT/visual['pdf']['path']
            if sha256(pdf)!=visual['pdf']['sha256'] or sha256(receipt_path)!=visual['export_receipt']['sha256']:
                raise RuntimeError('Reviewed development PDF changed')
            for output in receipt['outputs']:
                if sha256(ROOT/output['path'])!=output['sha256']:raise RuntimeError('Exported development document input changed')
            link='../../'+str(pdf.relative_to(ROOT));revision=receipt['identity']['revision']
            current_source=read_json(ROOT/'manuscript/manuscript_v04_inputs.json')['manuscript_sha256']
            source_notice=('The exported editable source matches the current manuscript.' if receipt['identity']['editable_source_sha256']==current_source else 'This is a preserved earlier PDF; newer editable-source evidence has not yet been exported and page-checked.')
            page=page.replace('<section><h2>English development manuscript</h2>',f'''<section><h2>Illustrated development PDF · revision {revision}</h2>
<p><a href="{link}">Open the {visual['pages_rendered']}-page article and figure supplement</a>. It contains {visual['existing_vector_figures_embedded']} existing source-development figures, including the actual decoder architecture, annotation-budget curves, classical controls, all21 selected UCF controls and all6 real held-source native galleries. {visual['copied_raw_case_supplements']} complete raw-eight-case PDF supplements accompany the article.</p>
<p class="small">All pages rendered; contact overviews and selected full pages actually inspected. {source_notice} Figures are appended for review, rather than final journal placement. This is an explicitly labelled development snapshot; independent confirmation remains pending. <a href="../../{str(visual_path.relative_to(ROOT))}">Render and visual-inspection receipt</a>.</p></section>
<section><h2>English development manuscript</h2>''')
    (BASE/'publication_progress.html').write_text(page)
    print(BASE/'publication_progress.html')


if __name__=='__main__':main()
