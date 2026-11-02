"""Build an English working manuscript from actual artifacts, without filling pending results."""
from pathlib import Path
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

from elresearch.common import ROOT,read_json,save_json,now,sha256
from publication_figures import publication_style, export_figure, render_architecture, render_heatmaps, render_model_comparison
publication_style()

FIG=ROOT/'artifacts/figures';FIG.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
budget=read_json(ROOT/'artifacts/budget_matrix.json')
rows=[]
for r in budget['runs']:
 rows.append({'encoder':r['encoder']['name'],'classifier':r['classifier'],'budget':r['budget'],'seed':r['seed'],
              'auc':r['validation_metrics']['auc'],'ap':r['validation_metrics']['average_precision'],
              'recall':r['validation_metrics']['recall'],'review':r['validation_metrics']['review_rate'],
              'n_train':len(r['training_ids'])})
data=pd.DataFrame(rows)
summary=data.groupby(['encoder','classifier','budget']).agg(auc_mean=('auc','mean'),auc_sd=('auc','std'),ap_mean=('ap','mean'),
                                                          recall_mean=('recall','mean'),review_mean=('review','mean')).reset_index()
summary.to_csv(ROOT/'artifacts/three_seed_summary.csv',index=False)
pretty={'dinov3_small':'DINOv3-S/16','resnet18':'ResNet-18','hog_stats':'HOG + statistics'}
classifier_name={'logistic':'Logistic','linear_svm':'Linear SVM','rbf_svm':'RBF SVM'}
palette={'dinov3_small':'#0072b2','resnet18':'#009e73','hog_stats':'#d55e00'}
markers={'dinov3_small':'o','resnet18':'s','hog_stats':'^'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(8.8,3.3),sharey=True)
for ax,clf in zip(axes,['linear_svm','rbf_svm']):
 for name in pretty:
  sub=summary[(summary.encoder==name)&(summary.classifier==clf)].sort_values('budget')
  x=sub.budget.to_numpy()*100;y=sub.auc_mean.to_numpy();s=sub.auc_sd.to_numpy()
  ax.plot(x,y,color=palette[name],marker=markers[name],ms=4,lw=1.5,label=pretty[name]);ax.fill_between(x,y-s,y+s,color=palette[name],alpha=.10,linewidth=0)
 ax.set(xscale='log',xticks=[5,10,20,100],xticklabels=['5','10','20','100'],ylim=(.48,1.005),
        xlabel='Labeled training images (%)',title=('(a) ' if clf=='linear_svm' else '(b) ')+classifier_name[clf]);ax.grid(axis='y',alpha=.2,linewidth=.5);ax.axhline(.5,color='#8a939b',lw=.7,ls='--')
axes[0].set_ylabel('Source validation AUROC');axes[1].legend(fontsize=8,loc='lower right')
fig.suptitle('Label efficiency on the fixed source-development split',fontsize=12)
fig.tight_layout();export_figure(fig,FIG/'three_seed_label_budget');plt.close(fig)

render_architecture()
render_heatmaps()

dense=[read_json(p) for p in (ROOT/'artifacts/dense_runs').glob('*/run.json')]
dense=[r for r in dense if r['budget']==1. and r['epochs']==40]
def dense_label(r):
 if r['architecture']=='deeplabv3_resnet50':return 'DeepLabv3 / R50 full tuning'
 return 'DINOv3 dense / '+ ('normal reference' if r['reference'] else r['adaptation'])
render_model_comparison(dense)
histories=[]
for p in (ROOT/'artifacts/dense_runs').glob('*/history.json'):
 started=read_json(p.parent/'started.json')
 if started['budget']==1. and started['epochs']==40:
  histories.append((started,read_json(p)))
fig,axes=plt.subplots(1,2,figsize=(8.8,3.7))
trace_colors={'deeplabv3_resnet50':'#0072b2','frozen':'#009e73','reference':'#d55e00','lora':'#cc79a7','full':'#62519f'}
for r,history in sorted(histories,key=lambda x:dense_label(x[0])):
 key='deeplabv3_resnet50' if r['architecture']=='deeplabv3_resnet50' else 'reference' if r['reference'] else r['adaptation']
 x=np.array([v['epoch'] for v in history]);y=np.array([v['validation_auc'] for v in history]);loss=np.array([v['loss'] for v in history])
 color=trace_colors[key];label=dense_label(r).replace('DINOv3 dense / ','DINOv3 ').replace('DeepLabv3 / R50 full tuning','DeepLabv3 / R50').replace('lora','LoRA')
 for ax,values in zip(axes,[y,loss]):
  ax.plot(x,values,color=color,lw=1.3,marker='o',ms=2.5,label=label,ls='--' if key=='full' else '-')
 best=np.argmax(y);axes[0].scatter(x[best],y[best],s=70,marker='*',color=color,edgecolor='white',linewidth=.5,zorder=5)
axes[0].set(xlabel='Epoch',ylabel='Validation AUROC',ylim=(.68,1.005),title='(a) Source validation')
axes[1].set(xlabel='Epoch',ylabel='Training objective',title='(b) Source training')
for ax in axes:ax.grid(axis='y',alpha=.2,linewidth=.5)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.025))
fig.tight_layout(rect=(0,.12,1,1));export_figure(fig,FIG/'dense_training_curves');plt.close(fig)

lit=read_json(ROOT/'references/literature.json');boot=read_json(ROOT/'artifacts/development_bootstrap.json')
full_boot=read_json(ROOT/'artifacts/full_development_bootstrap.json')['dense_local_to_reference']
uncertainty=[('DINOv3 vs ResNet Logistic',boot['resnet18_to_dinov3']),
             ('Shallow reference vs local',boot['local_to_reference']),
             ('Dense reference vs matched decoder',full_boot)]
fig,ax=plt.subplots(figsize=(8.8,2.8))
for i,(label,r) in enumerate(uncertainty):
 center=100*r['auc_delta'];lo,hi=np.array(r['paired_auc_difference_ci95'])*100
 ax.errorbar(center,i,xerr=[[center-lo],[hi-center]],fmt='o',capsize=3,color='#0072b2',ms=5,elinewidth=1.1)
ax.axvline(0,color='#b78350',linestyle='--',linewidth=1)
ax.set(yticks=range(3),yticklabels=[v[0] for v in uncertainty],xlabel='AUROC change (percentage points)',
       title='Conditional development uncertainty')
ax.invert_yaxis();ax.grid(axis='x',alpha=.15);fig.tight_layout()
export_figure(fig,FIG/'development_uncertainty');plt.close(fig)
manifest=read_json(ROOT/'data/ucf/manifest.json')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='PaperBody',fontName='Times-Roman',fontSize=10.7,leading=14,spaceAfter=7))
styles.add(ParagraphStyle(name='PaperTitle',fontName='Times-Bold',fontSize=20,leading=24,alignment=TA_CENTER,spaceAfter=14))
styles.add(ParagraphStyle(name='PaperHeading',fontName='Times-Bold',fontSize=14,leading=18,spaceBefore=10,spaceAfter=8,textColor=colors.HexColor('#17364c')))
styles.add(ParagraphStyle(name='PaperSmall',fontName='Times-Roman',fontSize=8.5,leading=11,spaceAfter=6))
styles.add(ParagraphStyle(name='PaperCell',fontName='Times-Roman',fontSize=8,leading=10))
story=[];sections=[]
def p(text,small=False):
 story.append(Paragraph(text,styles['PaperSmall' if small else 'PaperBody']));sections.append(text)
def h(text):story.append(Paragraph(text,styles['PaperHeading']));sections.append(text)
def table(rows,widths):
 rendered=[[Paragraph(str(v),styles['PaperCell']) for v in row] for row in rows]
 t=Table(rendered,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),
                       ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#718795')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d5dce1')),
                       ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
 story.append(t);story.append(Spacer(1,8))
def figure(path,width,caption):
 from PIL import Image as PILImage
 with PILImage.open(path) as image:height=width*image.height/image.width
 story.append(Image(str(path),width=width,height=height));p(caption,True)
def br():story.append(PageBreak())

story.append(Paragraph('From Frozen Features to Dense Models:<br/>A Source-Only Study of Photovoltaic Crack Inspection',styles['PaperTitle']))
p('<b>Working manuscript v0.3 - October 9, 2026.</b> Source-development results only. External confirmation, complete data-provenance resolution and full-network multi-seed experiments remain pending. This document is not a submission-ready claim of superiority.')
h('Abstract')
p('Electroluminescence inspection requires distinguishing annotated cracks from normal conductors, crystalline texture and other defects. We investigate a staged comparison of handcrafted descriptors, frozen pretrained representations and complete dense architectures. A pinned UCF pilot cohort contains 720 usable unique images, separated by conservative acquisition families into 430 training, 144 validation and 146 sealed test images. Frozen HOG, ImageNet ResNet-18 and DINOv3-S features are paired with three classical classifiers under four retrospective label budgets and three fixed seeds, producing 108 classifier runs. At full budget, CLS-based Logistic validation AUROC is 0.9495 for DINOv3 and 0.9252 for ResNet-18. A source-normal reference head achieves 0.9535 versus 0.9501 for its matched local baseline; the conditional group-bootstrap interval includes zero. We additionally construct and train full DeepLabv3/ResNet-50 and multi-depth DINOv3 dense networks, including LoRA and full-parameter variants. The complete DeepLab model has development image AUROC 0.9747, while the DINOv3 LoRA decoder has the highest crack pixel AP, 0.7098. We report these distinct outcomes without equating image AUROC to published pixel metrics. Source test and independent-domain outcomes are deliberately withheld until the protocol is frozen.')
p('<b>Keywords:</b> photovoltaic inspection; electroluminescence; vision foundation models; crack segmentation; normal references; source-only evaluation.')
h('1. Introduction and research questions')
p('A good image classifier need not localize a small crack, and a detector of unusual appearance need not distinguish cracks from non-crack defects. These distinctions matter when an inspection model changes laboratory, camera or module appearance. Our main task is the presence of an explicitly annotated crack, supplemented by native-mask localization. The ELPV expert-review task is kept separate: uncertainty-weighted judgments of functionality do not provide crack masks or calibrated physical severity [1].')
p('The study asks: (RQ1) how much improvement comes from pretrained image representations under matched classifiers and label budgets? (RQ2) does a complete dense model improve localization beyond frozen-vector classification? (RQ3) do similar normal structures provide additional evidence beyond identical pixel supervision? (RQ4) do these effects persist when thresholds and model settings are frozen before changing datasets? Only the source-development parts of these questions are answered by the current snapshot.')
p('The intended contribution is an audited benchmark and a test of conditional normal references, not the first use of deep features with SVM, a normal patch bank, or a transformer decoder. None of those ingredients alone establishes novelty [3,6,7,9].')
br()
h('2. Related work and legitimate comparisons')
p('The reference ELPV study compares local feature pipelines and a modified, staged-fine-tuned VGG-19 classifier [1]. Its strongest SVM uses KAZE/VGG features, so our HOG baseline must not be presented as a reproduction of its strongest method. Deep-feature extraction with classical classifiers and feature selection has also been studied on ELPV [3]. These works motivate a strong frozen-CNN comparator and a careful accounting of task definitions.')
p('The UCF data paper uses COCO-pretrained DeepLabv3/ResNet-50 with auxiliary supervision and five pixel classes [2]. Our full DeepLab model preserves that architecture family, while adapting it to a common three-class label map, whole-cell preprocessing, a grouped pilot cohort and a fixed optimizer. This is a controlled architecture comparator, not an exact reproduction of the original published score.')
p('PV-S3 uses labeled and unlabeled source images with a mean-teacher DeepLabv3+ framework [5]. Our current frozen-feature budgets use only the selected labels for fitting; they do not run semi-supervised training on the unselected images. AnomalyDINO establishes normal-patch nearest neighbors for industrial anomaly detection [6]. Our nearest-neighbor diagnostic uses DINOv3 and source pixel masks to select normal regions and is therefore not an equal-supervision reproduction. SuperADD already combines DINOv3 with local processing and a normal memory bank for industrial distribution shifts [9]. Conditional photovoltaic references therefore require evidence beyond those established ingredients.')
table([['Work','Published task and metric','Comparison status'],
       ['Deitsch et al. [1]','ELPV functional/defective classification; CNN accuracy 88.42%, SVM 82.44%','Context only; different task, split and sample weighting'],
       ['Demirci et al. [3]','ELPV two-class accuracy 94.52%; four-class 90.57%','Context only; no mRMR or exact DFB reproduction'],
       ['Fioresi et al. [2]','UCF five-class segmentation; pixel accuracy 95.4%, mIoU 57.3%','DeepLab architecture actually trained here; different label map and cohort'],
       ['PV-S3 [5]','UCF segmentation; reported mIoU 67.00% with 20% labels plus unlabeled source data','Full semi-supervised reproduction pending'],
       ['AnomalyDINO [6]','Industrial few-shot anomaly classification/localization','Conceptual diagnostic only; normal selection supervision differs']], [100,205,190])
p('Published accuracy, image AUROC, crack pixel AP and multi-class mIoU are not interchangeable. A numerical superiority claim would require a matched task, class map, split, supervision and evaluation rule. The literature table records reported values as context rather than placing them in the ranking of our runs.',True)
h('3. Data provenance and cohort construction')
p('The UCF publication describes 16,808 real cells parsed from 368 modules and 256 Griddler-Pro simulated cells, with simulation confined to training [2]. This addresses whether the original data are research-generated and peer-reviewed. It does not resolve the real/simulated identity of every file in the current public repository, whose inventory differs from the publication total. Accordingly, the pilot retains a provisional provenance flag.')
p('A label-blind hash ordering selected 768 pinned image/mask pairs. Forty-eight masks contain unexplained index 128 and were excluded rather than converted to normal. Exact pixel duplicates connect acquisition families before splitting. The 720 usable images use a single fixed grouped 60/20/20-style split. Physical module identities and crop associations still need verification. A fixed perceptual-hash and intensity-correlation screen found no candidates across UCF partitions or between UCF and SCDD under its stated tolerances; this finite screen is not proof that all overlapping origins or transformed crops are absent.')
br()
h('3.1. External data and supervision accounting')
table([['Dataset','Current usable scope','Role and restrictions'],['UCF pilot','720 images; 430 train / 144 validation / 146 test','Primary source development; test sealed; real/simulated file identity unresolved'],
       ['SCDD 20221008','588 confirmation candidates; parent 589-image pilot preserved','External confirmation candidate; no features, prototypes or model scores fitted on it'],
       ['ELPV','1968 original course-development images','Expert-review supplement only; 656 course test images remain sealed']], [95,180,220])
p('SCDD combines private laboratory and public sources [4]. Augmented derivatives and BMRK/SDLE public-source candidates were removed from the external scope to limit overlap. Its official class CSV maps normal structures, cracks and other defects; borders, text, padding and measurement artifacts are excluded from local supervision. The ambiguous star class is not silently treated as healthy. Release provenance supports real imagery, but module association, overlap and annotation exhaustiveness remain audit limitations. Two pixel-identical files have conflicting dense annotations (15,389 changed canonical pixels). The immutable confirmation child quarantines both identities, removing the one retained representative from the 589-image parent without resplitting any remaining origin.')
p('The public train/test CSVs list 9,562 and 2,289 files, matching the 11,851-pair training-tree inventory. They provide image/mask names but no explicit real/simulation identity field. The source-development validation cohort contains 58 crack-positive images and 86 no-crack images; all 86 negatives contain another annotated defect. Consequently, it does not estimate image-level false alarms on a wholly healthy-cell population.')
p('Each image with an explicit crack mask is positive. Other defects are difficult negatives for this binary task. Missing or uninterpretable annotation is unknown rather than normal. Dense supervision uses normal, crack and other-defect pixels; ignored regions remain ignored. The reference bank excludes any patch with a known non-normal pixel. Negative image labels rely on the completeness of the released annotations, not an independent physical inspection.')
p('Training budgets of 5%, 10%, 20% and 100% are retrospectively stratified by the released image label, with nested subsets for each seed. This is a benchmark sampling rule, not a prospective annotation acquisition procedure. Image-label and full-mask supervision are reported separately, and a fraction of selected images is not assumed to equal the same fraction of annotation time. All budgets retain the same labeled validation pool, which imposes an additional annotation-cost floor.')
h('3.2. Leakage controls and confirmatory protocol')
p('Preprocessing statistics used by pretrained backbones are fixed. Classifier scaling is fitted within source-training CV folds. Normal banks and loss class weights use only the selected source training images. Validation determines hyperparameters or stopping and the image threshold for at least 95% recall. No target-domain images are used to fit a classifier, a reference bank, normalization or a threshold. An evaluation command checks source-manifest, code and checkpoint checksums before opening a sealed domain and records an immutable opening ledger.')
p('Three seeds share the same source split and validation families. They assess subset selection, grouped-CV choices or optimization randomness; they are not three independent samples of the deployment population. Paired bootstrap resamples the provided acquisition families, not pixels or patches. Development intervals remain conditional on reused validation and do not constitute confirmatory tests.')
br()
h('4. Complete architectures and matched baselines')
figure(FIG/'full_architecture.png',495,'Figure 1. Implemented full architecture. The normal-reference slot is dimension-matched to a zero-residual baseline. A reference bank must be rebuilt after encoder adaptation; stale frozen references are prohibited during simultaneous LoRA training.')
p('Frozen image baselines use 1,806 HOG/statistical features, 512-dimensional ResNet-18 representations or 384-dimensional DINOv3-S CLS and valid-patch means. CLS, patch means and concatenation are compared in the initial full-budget experiment. Four-fold source grouped CV selects Logistic/Linear SVM C from 10^-5 to 10 and RBF C from {0.1,1,10}, with gamma in {0.1/d,1/d,10/d}. StandardScaler stays inside the CV pipeline.')
p('The full DINO network taps blocks 3, 6, 9 and 12, removes CLS/register offsets correctly, normalizes and projects each 21-by-21 patch grid to 128 channels, and fuses the depth maps. A 64-channel branch receives a dimension-matched normal-reference representation. Four convolutional stages upsample to 42, 84, 168 and 336 pixels while receiving raw normalized image detail at each scale. A three-class segmentation head and a CLS/mean/top-5%-evidence image head are jointly trained. This progressive decoder is a complete spatial network, although upsampling does not recover information absent in the original image.')
p('The conditional bank stores mask-supported normal patches from at most 256 source images. Global cosine similarity retrieves five reference images containing annotated normal patches, not necessarily globally healthy cells; local matching minimizes appearance distance plus 0.1 times squared normalized-coordinate distance. The local descriptor contains the query vector, its residual from the matched normal patch, two coordinates and appearance distance. Training excludes the query acquisition family from the bank.')
br()
h('4.1. Optimization and adaptation')
p('DeepLabv3/ResNet-50 uses the official COCO-with-VOC pretrained weights, its ASPP and auxiliary classifier, and three adapted output classes. All affine weights are trainable; pretrained BatchNorm running statistics remain fixed for small local batches. The DINO decoder supports a frozen encoder, rank-8 query/value LoRA in the final four blocks, or full parameter tuning. The fixed initial four-architecture comparison includes DeepLab full tuning, frozen DINO without/with normal references, and DINO LoRA without a stale reference bank. A separately recorded, subsequent source-development extension trains all DINO encoder and decoder parameters under the same optimization rules; it is not an independently confirmed experiment.')
p('Both dense families receive the same primary image and canonical pixel supervision and the same image BCE + weighted pixel CE + 0.5 crack Dice objective. DeepLab additionally retains its standard auxiliary pixel CE with weight 0.4. Training uses AdamW, learning rates 10^-4 for backbone/LoRA and 10^-3 for heads, batch size 2, up to 40 epochs and patience 6. Source image AUROC selects the checkpoint. The initial architecture pilot uses no augmentation, which limits conclusions relative to published augmented training. All validation epochs are retained.')
p('The backbone learning rate is a single fixed pilot setting rather than an adaptation-specific optimization search. A poor full-tuning result under this setting cannot establish the inferiority of full tuning after adequate optimization. Additional learning-rate or augmentation choices must be developed on source data before independent-domain confirmation.')
p('The same image is letterboxed rather than center-cropped, gray is copied into RGB, and ImageNet normalization is applied. Masks follow the same geometry; an annotated thin crack is preserved under downsampling. This design differs from the published 224-pixel random crops [2,5]. ResNet, COCO and DINOv3 pretraining data/objectives differ, so comparisons estimate whole-system performance rather than isolating architecture alone.')
h('5. Source-development results')
full=summary[summary.budget==1.]
t=[['Frozen descriptor','Classifier','AUROC mean +/- SD','AP mean','Recall / review rate']]
for name in pretty:
 for clf in classifier_name:
  row=full[(full.encoder==name)&(full.classifier==clf)].iloc[0]
  t.append([pretty[name],classifier_name[clf],f'{row.auc_mean:.4f} +/- {row.auc_sd:.4f}',f'{row.ap_mean:.4f}',f'{row.recall_mean:.3f} / {row.review_mean:.3f}'])
table(t,[105,80,115,65,130])
p('Table 2. Three seeds at 100% image-label budget on the fixed 144-image validation partition. The descriptor is CLS for pretrained encoders. Zero standard deviation can occur when different CV seeds select the same deterministic estimator; it does not imply zero population uncertainty.',True)
comparison=boot['resnet18_to_dinov3'];ci=comparison['paired_auc_difference_ci95']
p(f'The seed-11 DINOv3-versus-ResNet Logistic AUROC difference is {comparison["auc_delta"]:.4f}; its conditional paired 95% family-bootstrap interval is [{ci[0]:.4f}, {ci[1]:.4f}]. The interval includes zero. Frozen DINOv3 clearly improves on the handcrafted pilot, but the stronger CNN comparator narrows the observed gap. This motivates independent-domain confirmation rather than a claim of established superiority.')
br()
h('5.1. Label-budget behavior')
figure(FIG/'three_seed_label_budget.png',495,'Figure 2. Identical classifiers isolate descriptor-family comparisons across retrospective labeled-image budgets. Shading is one SD across three seeds, not a sampling confidence interval; all points share the same validation families.')
p('The complete 108-run matrix comprises three descriptors, three classifiers, four budgets and three seeds. It does not add target-domain fitting or unused-image semi-supervised losses. The initial representation ablation is complementary and is not pooled into the seed counts. Selected source image IDs, CV searches, raw predictions and threshold records accompany each saved run. The advantage is budget-dependent: at 5% labels, ResNet-18 exceeds DINOv3 with Linear SVM in this pilot, so the full-budget improvement does not establish uniform label efficiency.')
figure(FIG/'development_uncertainty.png',495,'Figure 3. Paired 95% intervals from 2,000 acquisition-family bootstrap draws, conditional on the reused validation cohort and selected models. All three intervals include zero. They do not account for checkpoint selection or substitute for independent-domain confirmation.')
br()
h('5.2. Complete dense-model comparison')
t=[['Architecture','Trainable / total params','Validation AUROC','Native pixel AP / positive Dice']]
for r in sorted(dense,key=dense_label):
 m=r['validation_localization'];t.append([dense_label(r),f'{r["trainable_parameters"]/1e6:.2f}M / {r["total_parameters"]/1e6:.2f}M',f'{r["validation_metrics"]["auc"]:.4f}',f'{m["pooled_pixel_AP"]:.4f} / {m["mean_Dice_positive_images"]:.4f}'])
if not dense:t.append(['Full training in progress','Do not rank smoke checks','Pending','Pending'])
table(t,[150,110,85,150])
p(f'Table 3. Completed full-budget, seed-11, 40-epoch-cap runs present in this snapshot: {len(dense)}/5 (four initial architectures plus one full-tuning extension). One-epoch, 10%-budget architecture smoke checks are excluded from the ranking. Full-network results are checkpoint-selected development values; native overlap uses fixed crack probability 0.5. A three-class comparison is not the original five-class UCF paper metric.',True)
p('The appended full-parameter DINOv3 experiment trains all 22.98 million encoder/decoder parameters and reaches AUROC 0.9583, pixel AP 0.6275 and positive-image Dice 0.3719. It does not improve on the frozen or LoRA variants under this fixed, small-cohort optimization protocol. This result is retained rather than suppressed; it does not establish that full tuning is generally ineffective.')
p('DeepLab has the highest development image AUROC (0.9747), whereas DINOv3 plus LoRA has the highest pooled crack pixel AP (0.7098) and positive-image Dice (0.4836). LoRA does not improve image AUROC over the frozen decoder in this run. Normal references raise frozen-decoder AUROC from 0.9687 to 0.9717 and positive-image Dice from 0.4201 to 0.4666, but all-image Dice falls from 0.7664 to 0.7435. This tradeoff prevents a claim of uniform reliability improvement.')
fullboot=read_json(ROOT/'artifacts/full_development_bootstrap.json')['dense_local_to_reference'];fullci=fullboot['paired_auc_difference_ci95']
p(f'The full-decoder reference AUROC change is {fullboot["auc_delta"]:.4f}, with conditional paired 95% family-bootstrap interval [{fullci[0]:.4f}, {fullci[1]:.4f}]. Checkpoint selection and thresholding reused the same validation set, so this descriptive interval does not remove model-selection optimism.')
figure(FIG/'dense_training_curves.png',495,'Figure 4. Validation AUROC and source-training objectives, with stars marking AUROC-selected checkpoints. All epochs are retained. The DeepLab auxiliary loss differs, so training objective values are descriptive rather than a common performance ranking.')
br()
h('6. Local evidence and failure analysis')
figure(FIG/'full_validation_examples.png',495,'Figure 5. Predictions in the original EL image context. Color represents crack probability on the fixed 0-1 scale; overlay opacity is 0.78 times probability. Cyan solid contours are annotated cracks; orange dashed contours are predictions above the fixed pixel threshold 0.5. White boxes indicate the companion crops. Image decisions use each saved source-validation threshold. These are the original fixed development cases, including misses, not newly selected success examples.')
p('Contextual overlays retain the original cell texture, conductors and brightness rather than replacing the image with a black probability background. They are segmentation outputs, not transformer attention maps. Display opacity changes visibility only; it does not change stored predictions or quantitative scores. The companion opaque maps allow probability values to be read without a blended background.')
p('Heatmaps display interpolated model responses. They do not establish understanding of crack physics, prove electrically inactive area, or reconstruct lost spatial detail. The validation subset is too small and too reused to select a new method based on appealing examples. Normal-structure false positives and small-crack misses must be quantified on predeclared subsets after provenance and domain freezing.')
br()
h('6.1. Native-coordinate local evidence')
figure(FIG/'full_validation_zoom.png',495,'Figure 6. Identical native-coordinate crops for all models, with opaque crack probabilities on the same 0-1 color scale. Positive crops use annotation extent plus context; the negative crop is the fixed central third of the image. No per-panel rescaling or additional smoothing is applied. Cached maps use float16, so very small values may be rounded to zero. A magnified crop improves readability and does not recover new image detail.')
local=boot['local_to_reference'];ci=local['paired_auc_difference_ci95']
p(f'The shallow normal-reference versus matched local-head AUROC difference is {local["auc_delta"]:.4f}, with a conditional 95% family-bootstrap interval [{ci[0]:.4f}, {ci[1]:.4f}]. The interval includes zero. A high image score does not validate localization: in the smallest annotated example, neither coarse head assigns a substantial response to the annotated region. The 39-pixel annotation in Figures 5 and 6 remains almost unresponsive in the complete models. An improved average metric therefore does not demonstrate that tiny cracks have become detectable.')
failures=read_json(ROOT/'artifacts/dense_failure_subsets.json')
q=failures['training_positive_quartiles'][0]
p(f'A descriptive post-hoc small-region subset uses the lower quartile of crack area fraction in source training ({q:.4f}); only eight validation positives meet this rule. It measures annotated region size, not physical crack severity. At the saved high-recall image thresholds, classification recall and native Dice must be inspected separately. The subset uses cached float16 native maps and is not a confirmatory subgroup test.')
br()
h('6.2. Quantitative failure diagnostics')
t=[['Full architecture','Small subset recall / Dice','No-crack false-positive rate','Normal-pixel false-positive rate']]
for r in sorted(failures['runs'],key=dense_label):
 z=r['subsets'];t.append([dense_label(r),f'{z["small_crack"]["image_positive_rate"]:.3f} / {z["small_crack"]["mean_crack_Dice"]:.3f}',f'{z["no_crack"]["image_positive_rate"]:.3f}',f'{r["pooled_normal_pixel_false_positive_rate"]:.5f}'])
table(t,[155,115,115,110])
p('Table 4. Post-hoc source-validation diagnostics. No-crack negatives contain other defects; this image-level false-positive rate is not a healthy-cell false-alarm estimate. Normal-pixel error uses only class-0 pixels, with fixed crack probability 0.5. Pixel counts do not provide independent statistical replicates.',True)
h('6.3. Warm inference and checkpoint restoration')
benchmarks=[]
for r in dense:
 path=ROOT/'artifacts/dense_runs'/r['run_id']/'benchmark.json'
 if path.exists():benchmarks.append((r,read_json(path)))
t=[['Architecture','Warm median ms / image','Model / normal bank size']]
for r,b in sorted(benchmarks,key=lambda x:dense_label(x[0])):
 t.append([dense_label(r),f'{1000*b["median_seconds_per_image"]:.2f}',
           f'{b["checkpoint_file_bytes"]/1e6:.1f} MB / {b["bank_file_bytes"]/1e6:.1f} MB'])
table(t,[220,130,145])
p('Table 5. Mac M2 Max, float32 MPS, batch size two, five warm-up batches and ten timed batches in a separate process per model. The same two source-validation images are decoded, checked, resized, encoded, classified and reprojected on each iteration; the reference version also performs global retrieval and local matching. Device synchronization brackets timing. Model/bank load and output-file serialization are excluded. This two-image, warm-cache measurement is not whole-dataset or deployment throughput.',True)
p('All five complete checkpoints are strictly restored without downloading substitute weights; reloaded image predictions agree with saved validation predictions within the declared numerical tolerance. MPS after-batch allocator and driver usage are recorded separately from latency. They are not peak CUDA memory estimates; deployment GPU peak-memory measurement remains pending.')
br()
h('7. Limitations and completion criteria')
p('The current cohort is a pilot subset, source-file real/simulated identities are not fully resolved, and grouping is conservative rather than confirmed physical-module identity. Exact duplicate checks do not exhaust near duplicates, shared crops or acquisition linkage. Released annotations may miss faint cracks; their normal background is not independent physical verification. Pretraining-corpus membership is not auditable, and the compared pretrained systems have unequal data and objectives. Single-seed dense results, selected on the same validation data used for thresholding, are especially vulnerable to selection optimism.')
p('Pixel supervision is more informative than image labels. Adding a dense head therefore changes both architecture and supervision relative to an image-only SVM. Only matched dense/local ablations can support an attribution to normal references. A three-class canonical label map cannot reproduce five-class mIoU, and semi-supervised PV-S3 has access to unselected source images that our current classifiers do not exploit. Native positive-image Dice and all-image Dice must both be inspected to avoid apparent gains from empty normal masks.')
p('Before confirmatory evaluation, expand and audit the source cohort while preserving previously assigned and observed origins; resolve synthetic identities, duplicate/crop associations and label ambiguities; repeat full-network seeds and budgets; quantify native-mask localization, normal-structure errors and runtime/memory; fix models, code, thresholds and comparison hypotheses. Then evaluate source test, SCDD transfer, reverse transfer and a separately audited PVEL confirmation domain. Any later target-domain adaptation must be reported as a separate protocol. Source test, SCDD and course test scores are not supplied in this manuscript.')
h('8. Current conclusion')
p('Frozen pretrained features are a strong starting point for solar-cell crack classification, and a full spatial architecture is now implemented rather than assumed from an embedding experiment. The normal-reference hypothesis remains uncertain: its initial gain is small and its development interval crosses zero. The evidence supports continuing matched full-model, low-label and independent-domain experiments. It does not yet establish a publishable method advantage or reliable cross-domain detection.')
br()
h('References')
order=['Deitsch2019','Fioresi2022','Demirci2021','Pratt2023','Jha2025','Damm2025','Simeoni2025','Hu2022','Roming2026']
for i,key in enumerate(order,1):
 r=next(x for x in lit if x['key']==key)
 p(f'[{i}] {html.escape(r["authors"])}. {html.escape(r["title"])}. {html.escape(r["venue"])}, {r["year"]}. <link href="{html.escape(r["url"],quote=True)}" color="#235e83">Primary reference</link>.',True)

def page(canvas,doc):
 canvas.saveState();canvas.setFont('Times-Roman',8);canvas.setFillColor(colors.HexColor('#607180'))
 canvas.drawString(50,A4[1]-27,'Photovoltaic crack inspection - working manuscript / source-development evidence')
 canvas.drawString(50,26,'v0.3 | No confirmatory test or cross-domain score is claimed')
 canvas.drawRightString(A4[0]-50,26,str(doc.page));canvas.restoreState()
destination=OUT/'research_manuscript_v03.pdf'
SimpleDocTemplate(str(destination),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=45,
                  title='From Frozen Features to Dense Models: Photovoltaic Crack Inspection',author='Research working draft').build(story,onFirstPage=page,onLaterPages=page)
save_json(ROOT/'output/manuscript_source.json',{'created_at':now(),'sections':sections,'literature':lit,'full_dense_completed':len(dense),
                                              'artifact_inputs':{'budgets':sha256(ROOT/'artifacts/budget_matrix.json'),'bootstrap':sha256(ROOT/'artifacts/development_bootstrap.json')}})
full_table=pd.DataFrame([{'architecture':dense_label(r),'run_id':r['run_id'],'auc':r['validation_metrics']['auc'],
                         **r['validation_localization'],'trainable_parameters':r['trainable_parameters']} for r in dense])
full_table.to_csv(ROOT/'artifacts/full_dense_results.csv',index=False)
display_dense=full_table[['architecture','auc','pooled_pixel_AP','mean_Dice_positive_images',
                          'mean_Dice_all_images','trainable_parameters']].rename(columns={
 'auc':'Image AUROC','pooled_pixel_AP':'Crack pixel AP','mean_Dice_positive_images':'Positive-image Dice',
 'mean_Dice_all_images':'All-image Dice','trainable_parameters':'Trainable parameters'})
from render_research_report import render_report
render_report()
print({'manuscript':str(destination),'complete_dense_runs':len(dense),'classifier_runs':len(budget['runs'])})
