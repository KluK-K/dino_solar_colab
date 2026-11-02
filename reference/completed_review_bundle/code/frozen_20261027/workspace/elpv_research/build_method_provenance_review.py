"""Traceable, bounded method-attribution review; no model or final-data access."""
import html
from pathlib import Path

from elresearch.common import ROOT, now, read_json, save_json, sha256

SOURCE = ROOT / 'references/publication_20261017/method_provenance_review_20261018'
OUTPUT = ROOT / 'artifacts/publication_20261017/method_provenance_v1'


RELATED = r'''\section{Related methods and scope of comparison}
HOG~\cite{hog} and support-vector classification~\cite{svm} motivate the handcrafted and frozen-feature controls. ResNet~\cite{resnet} and EfficientNet~\cite{efficientnet} supply convolutional baselines; DeepLabv3+~\cite{deeplab} supplies the segmentation architecture family. Our HOG/statistics fusion, modified task heads and adapted PS-MT architecture are specified separately. The original method papers establish attribution, not EL performance in our cohort. LoRA~\cite{lora} parametrizes weight changes with low-rank matrices while freezing the original weights. It changes the encoder, whereas a new classifier learns a mapping from its output. Its original language-model results do not establish a photovoltaic advantage. Temperature scaling~\cite{calibration} supplies source-fitted probability calibration, without guaranteeing calibration after acquisition shift.

Local nominal-feature comparison is established prior work. PatchCore~\cite{patchcore} combines intermediate pretrained CNN features, local aggregation, representative coreset memory and nearest-neighbor scoring. AnomalyDINO~\cite{anomalydino} uses frozen DINOv2 patch features, cosine-neighbor distances, reference rotations and optional feature-derived foreground masking. Its default image statistic averages the largest 1\% of patch distances. We review that paper as prior art; DINOv2 is not downloaded or executed in this study.

SuperADD~\cite{superadd} already combines DINOv3 with nominal-feature memory under acquisition variation. Its public workshop-challenge technical report describes a ViT-H+/16 backbone, four intermediate feature banks, overlapping 640-pixel processing windows, intensity augmentation, memory subsampling and morphological postprocessing. It is distinct from a CVPR main-conference publication. Consequently, neither DINOv3-based industrial transfer nor DINOv3 plus local normal memory is claimed as our invention.

Our registered experiment instead tests crack-specific, mask-informed normal references within supervised global/local prediction, with matched annotation identities and explicit source exposure. Its global retrieval, position-regularized residual input and learned decoder differ from the training-free pipelines above. A distance-only control is an adaptation of the nominal-neighbor principle, not an exact reproduction of PatchCore, AnomalyDINO or SuperADD. Other annotated defects are difficult crack-negative cases here but may be anomalies under those papers' tasks. Their MVTec/VisA scores therefore cannot rank against our EL scores. The intended contribution is the evidence on label requirements, native crack localization, source-frozen decisions and acquisition transfer, including failures. Existing PV-S3/PS-MT methods receive separately documented supervised and semi-supervised comparisons under our task protocol; no published headline score is presented as a matched result.

'''

BIBLIOGRAPHY = r'''\bibitem{hog} N. Dalal and B. Triggs. Histograms of Oriented Gradients for Human Detection. \emph{CVPR},2005. \url{https://doi.org/10.1109/CVPR.2005.177}.
\bibitem{svm} C. Cortes and V. Vapnik. Support-vector networks. \emph{Machine Learning},20:273--297,1995. \url{https://doi.org/10.1007/BF00994018}.
\bibitem{resnet} K. He,X. Zhang,S. Ren,and J. Sun. Deep Residual Learning for Image Recognition. \emph{arXiv:1512.03385},2015. \url{https://arxiv.org/abs/1512.03385}.
\bibitem{efficientnet} M. Tan and Q. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. \emph{ICML},PMLR97:6105--6114,2019. \url{https://proceedings.mlr.press/v97/tan19a.html}.
\bibitem{deeplab} L.-C. Chen,Y. Zhu,G. Papandreou,F. Schroff,and H. Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. \emph{ECCV},2018. \url{https://doi.org/10.1007/978-3-030-01234-2_49}.
\bibitem{lora} E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. \emph{arXiv:2106.09685},2021. \url{https://arxiv.org/abs/2106.09685v2}.
\bibitem{calibration} C. Guo,G. Pleiss,Y. Sun,and K. Q. Weinberger. On Calibration of Modern Neural Networks. \emph{ICML},PMLR70:1321--1330,2017. \url{https://proceedings.mlr.press/v70/guo17a.html}.
\bibitem{patchcore} K. Roth et al. Towards Total Recall in Industrial Anomaly Detection. \emph{CVPR},2022. \url{https://arxiv.org/abs/2106.08265v2}.
\bibitem{anomalydino} S. Damm,M. Laszkiewicz,J. Lederer,and A. Fischer. AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2. \emph{WACV},2025. \url{https://arxiv.org/abs/2405.14529v3}.
\bibitem{superadd} L. Roming et al. SuperADD: Training-free Class-agnostic Anomaly Segmentation---CVPR2026 VAND4.0 Workshop Challenge Industrial Track. \emph{arXiv:2605.14808v1},2026. Workshop-challenge technical report. \url{https://arxiv.org/abs/2605.14808v1}.
'''


def record(key, title, source, access, mechanism, regime, relevance, limitation, pages=None):
    return dict(key=key, title=title, primary_source=source, evidence_access=access,
                mechanism=mechanism, task_or_regime=regime, relevance=relevance,
                applicability_limit=limitation, reviewed_pages=pages or [],
                original_paper_score_used_in_our_results=False)


def main():
    access = read_json(SOURCE / 'fulltext_access.json')
    for row in access['records']:
        for key in ['pdf', 'text']:
            if sha256(ROOT / row[key]) != row[key + '_sha256']:
                raise RuntimeError('Original full-text snapshot changed: ' + row['name'])
    records = [
        record('hog','Histograms of Oriented Gradients for Human Detection',
            'https://doi.org/10.1109/CVPR.2005.177','Original DOI metadata only',
            'HOG feature-family attribution only.','Original human-detection method.',
            'Our controls use HOG plus42 statistical features, not the original detector.',
            'Metadata does not constitute a full-text method or performance review.'),
        record('svm','Support-vector networks','https://doi.org/10.1007/BF00994018',
            'Original publisher abstract/bibliography; full text subscription-limited',
            'Feature-space decision-surface classification with nonseparable examples.',
            'Original two-group classification work.',
            'Attribute the support-vector classifier; our grids/preprocessing are explicit.',
            'No original optical-character-recognition score is a PV comparison.'),
        record('resnet','Deep Residual Learning for Image Recognition',
            'https://arxiv.org/abs/1512.03385','Original abstract/bibliography only in this review',
            'Residual-function network architecture.','Natural-image recognition.',
            'ResNet18/50 baselines and the deep-stem PS-MT backbone are separately specified.',
            'Architecture attribution does not verify our library weights or EL performance.'),
        record('efficientnet','EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks',
            'https://proceedings.mlr.press/v97/tan19a.html','Original PMLR abstract/bibliography only',
            'Balanced depth, width and input-resolution scaling of convolutional networks.',
            'Natural-image classification and transfer.',
            'EfficientNet-B0 is a CNN comparator, with its own source optimization checks.',
            'Original benchmark efficiency does not measure our end-to-end EL cost.'),
        record('deeplab','Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation',
            'https://arxiv.org/abs/1802.02611v3','Selected original full-text method sections read',
            'Atrous contextual encoder/ASPP and a spatial-detail decoder.',
            'Supervised semantic segmentation on VOC/Cityscapes.',
            'Our adapted baseline and exact PS-MT deep-stem port identify distinct implementations.',
            'Three EL classes, masks, losses, resolution and training differ from the original paper.',
            [3,4,5]),
        record('lora','LoRA: Low-Rank Adaptation of Large Language Models',
            'https://arxiv.org/abs/2106.09685v2','Selected original full-text method sections read',
            'Freeze base weights; train scaled low-rank BA updates; merged weights can be used for inference.',
            'Transformer language-model adaptation in the original empirical study.',
            'Our rank8 last-four-block Q/V adaptation is distinguished from a classifier on fixed vectors.',
            'Original NLP gains or inference claims do not establish our unmerged/merged full-pipeline cost.',
            [4,5]),
        record('patchcore','Towards Total Recall in Industrial Anomaly Detection',
            'https://arxiv.org/abs/2106.08265v2','Selected original full-text method sections read',
            'Locally pooled intermediate CNN features; greedy projected coreset; Euclidean neighbors and reweighted image score.',
            'Nominal-only industrial anomaly detection, chiefly MVTec AD.',
            'Direct antecedent of normal patch memory; compare supervision, memory and scoring explicitly.',
            'Our capped cosine memory is not its coreset or exact scoring; no PatchCore reproduction claim.',
            [3,4,5]),
        record('anomalydino','AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2',
            'https://arxiv.org/abs/2405.14529v3','Selected original full-text methods/evaluation read; WACV2025 attribution verified',
            'Frozen DINOv2 patches; cosine nearest normal neighbor; top1% mean image score; rotations and optional PCA-derived masking.',
            'Training-free nominal few-shot MVTec AD/VisA detection and localization.',
            'Direct precedent for frozen transformer normal memory; label-free nominal references differ from our mask-informed selection.',
            'Read/cited only; no DINOv2 weights or execution. Source thresholds/top5% supervised fusion are different.',
            [1,3,4,5,6]),
        record('superadd','SuperADD: Training-free Class-agnostic Anomaly Segmentation',
            'https://arxiv.org/abs/2605.14808v1','Selected original full-text methods/evaluation read; workshop-challenge technical report',
            'DINOv3-H+/16 four-depth memory; overlapping640-pixel windows, intensity scaling, feature subsampling and morphological masks.',
            'Nominal-only MVTec AD2 acquisition/illumination changes; one configuration across classes.',
            'Explicit prior DINOv3 normal-memory system; rules out claiming that broad combination as new.',
            'Different task/model/input/normal data/postprocessing; not an exact baseline run or main-conference publication.',
            [1,2,3,4]),
        record('calibration','On Calibration of Modern Neural Networks',
            'https://proceedings.mlr.press/v70/guo17a.html','Original PMLR abstract/bibliography only',
            'Temperature scaling as a single-parameter post-processing calibration method.',
            'Image/document classifiers in the original empirical evaluation.',
            'Attribute source-only temperature fitting; report target Brier and ranking separately.',
            'No blanket guarantee under acquisition shift; our development counterexample is separate evidence.')]
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'related_work.tex').write_text(RELATED)
    (OUTPUT / 'bibliography_additions.tex').write_text(BIBLIOGRAPHY)
    bindings = [SOURCE/'review_scope.json',SOURCE/'fulltext_access.json',SOURCE/'metadata_page_snapshots.json',
                SOURCE/'hog_metadata.json',SOURCE/'deeplab_metadata.json',SOURCE/'svm_metadata.json',SOURCE/'resnet_metadata.json']
    for row in access['records']:
        bindings += [ROOT/row['pdf'],ROOT/row['text']]
    metadata = read_json(SOURCE/'metadata_page_snapshots.json')
    bindings += [ROOT/row['path'] for row in metadata['records'] if 'path' in row]
    bindings += [ROOT/'elresearch/publication_dense_models.py',ROOT/'elresearch/mechanistic_models.py',
                 ROOT/'elresearch/publication_normal_bank.py',ROOT/'elresearch/classical_features.py',
                 OUTPUT/'related_work.tex',OUTPUT/'bibliography_additions.tex',Path(__file__).resolve()]
    review = dict(at=now(),question=read_json(SOURCE/'review_scope.json')['question'],
        records=records,systematic_exhaustive_review=False,
        metadata_failures_retained=['Crossref429: SVM and ResNet; original publisher/arXiv independently inspected'],
        research_model_or_dataset_downloaded=False,DINOv2_download_or_execution=False,
        final_confirmation_access=False,scope='Method attribution and selected-section comparison, not a systematic novelty certificate or new experimental result.',
        input_bindings=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in bindings],
        manuscript_fragment=dict(path=str((OUTPUT/'related_work.tex').relative_to(ROOT)),sha256=sha256(OUTPUT/'related_work.tex')),
        bibliography_fragment=dict(path=str((OUTPUT/'bibliography_additions.tex').relative_to(ROOT)),sha256=sha256(OUTPUT/'bibliography_additions.tex')))
    save_json(OUTPUT/'review.json',review)
    esc=html.escape
    rows=''.join('<tr>'+''.join('<td>'+v+'</td>' for v in [
        '<a href="'+esc(r['primary_source'])+'">'+esc(r['title'])+'</a><br>'+esc(r['evidence_access'])+
        ('<br>Read PDF pages: '+', '.join(map(str,r['reviewed_pages'])) if r['reviewed_pages'] else ''),
        esc(r['mechanism'])+'<p>'+esc(r['task_or_regime'])+'</p>',
        esc(r['relevance'])+'<p>'+esc(r['applicability_limit'])+'</p>'])+'</tr>' for r in records)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Primary-method provenance</title>
<style>body{font:16px/1.55 system-ui;margin:36px auto;max-width:1260px;padding:0 24px;color:#19343e}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%}td,th{padding:16px;text-align:left;vertical-align:top;border-bottom:1px solid #cddadd;width:33%}th{background:#eaf2f3}a{color:#216285}.note{background:#f2f5f5;padding:16px;border-left:4px solid #328785}small{color:#51676f}</style>
<h1>What is inherited, and what our EL study tests</h1>
<p class="note">DINOv3 plus a local nominal-feature bank is existing prior art. Our contribution remains an empirical hypothesis about crack-specific reliability under limited annotation and acquisition shift. Independent confirmation is still pending.</p>
<p>This bounded primary-source review distinguishes selected full-text sections from abstract/metadata access. It introduces no new model family and makes no cross-paper score ranking. All scientific workers and sealed confirmation outcomes remain unchanged.</p>
<table><tr><th>Original work and access</th><th>Mechanism and original regime</th><th>Relation to our implementation and limits</th></tr>'''+rows+'''</table>
<h2>Required evidence for a contribution</h2><p>Use matched original annotation identities; compare global, supervised local, normal-distance and reference-conditioned systems; retain both improved and adverse endpoints. Report source-selection exposure, native localization, source-frozen recall/review and complete cost. If the reference mechanism does not beat the stronger controls, report that boundary instead of relabelling the combination as novel.</p>
<p><small>Original PDFs, extraction hashes, observed metadata failures and inspected implementation hashes are recorded in the local review JSON. PDF download alone is not called a full-text review; only listed sections were inspected. Workshop technical reports and peer-reviewed conference papers retain distinct status.</small></p></html>'''
    (OUTPUT/'report.html').write_text(page)
    print('Primary-method review:',len(records),'works;5 selected-full-text,4 abstract/bibliographic,1 metadata-only; no new experiment')


if __name__ == '__main__':
    main()
