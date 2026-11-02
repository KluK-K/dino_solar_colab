"""Targeted attribution for five components already used in registered code.

No scientific source edits, model imports, fitting, inference or target access.
Primary papers are read locally; discovery errors remain in the source records.
"""
from pathlib import Path
import hashlib
import html
import json
from datetime import datetime, timezone

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
REFS = ROOT / 'references/publication_20261017/component_method_review_20261026'
OUT = BASE / 'component_method_provenance_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'review.json').exists(), 'Retain the completed review unchanged'
    assert not (BASE / 'final_confirmation_v1/observation_opened.json').exists()
    state = json.loads((BASE / 'inference_efficiency_v1/status.json').read_text())['state']
    assert state.startswith('queued'), 'Do not compete with formal timings'
    records = [
        {'key': 'adamw', 'citation': 'I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. ICLR, 2019.',
         'primary_source': 'https://arxiv.org/abs/1711.05101v3', 'reviewed_PDF_pages': [1, 2, 3],
         'mechanism': 'AdamW separates parameter decay from the adaptive gradient update; Algorithm2 describes the distinction from adding an L2 term to the gradient.',
         'implemented_use': 'Existing image-only and joint dense training construct torch.optim.AdamW with registered encoder/head rates and weight decay.',
         'code': ['elresearch/publication_train.py', 'elresearch/publication_dense_train.py'],
         'limit': 'Attribution does not reproduce original benchmark scores, equate AdamW and SGD objectives, or establish an EL or runtime advantage.'},
        {'key': 'mean_teacher', 'citation': 'A. Tarvainen and H. Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. NeurIPS30, 2017.',
         'primary_source': 'https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html',
         'reviewed_PDF_pages': [1, 2, 3],
         'mechanism': 'Teacher targets use an exponential moving average of student weights, updated after student steps; the original work commonly uses prediction MSE.',
         'implemented_use': 'The PS-MT/PV-S3 port uses two teachers. Active-teacher identity alternates by round, but EMA runs after every student optimizer update in that round. Floating buffers are averaged and integer counters copied.',
         'code': ['elresearch/published_train.py', 'elresearch/published_semisup.py'],
         'limit': 'The port has confidence-weighted CE, separate random teacher initialization and two-teacher assistance. It is not the original single-teacher MSE experiment.'},
        {'key': 'cutmix', 'citation': 'S. Yun et al. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. ICCV, 2019.',
         'primary_source': 'https://arxiv.org/abs/1905.04899v2', 'reviewed_PDF_pages': [1, 3],
         'mechanism': 'The original method replaces rectangular image regions and mixes classification labels by area; its experiments set the symmetric Beta parameter to1.',
         'implemented_use': 'Existing source-only SSL mixing replaces corresponding image regions, dense teacher logits and geometric-validity masks, with registered Beta(4,4) sampling and batch donors.',
         'code': ['elresearch/published_data.py', 'elresearch/published_train.py'],
         'limit': 'The segmentation/pseudo-logit adaptation and Beta(4,4) are explicit port settings, not the original classification recipe or evidence of its EL benefit.'},
        {'key': 'vat', 'citation': 'T. Miyato, S.-i. Maeda, M. Koyama and S. Ishii. Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning. Author manuscript, arXiv:1704.03976v2, 2018 revision.',
         'primary_source': 'https://arxiv.org/abs/1704.03976v2', 'reviewed_PDF_pages': [1, 3, 4, 5],
         'mechanism': 'VAT penalizes local prediction-distribution changes without requiring the true label for the perturbation direction; the paper describes KL, finite differences and power-iteration approximations.',
         'implemented_use': 'The port obtains a normalized one-step feature perturbation from two frozen teacher decoders, uses registered xi/epsilon and differentiates through perturbed student decoding.',
         'code': ['elresearch/published_semisup.py', 'elresearch/published_train.py'],
         'limit': 'The encoder-feature/two-teacher implementation follows the reviewed PS-MT/PV-S3 port rather than reproducing the original input-space VAT experiment. Existing literal-xi sensitivity stays unchanged.'},
        {'key': 'vnet', 'citation': 'F. Milletari, N. Navab and S.-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV, 2016. DOI:10.1109/3DV.2016.79.',
         'primary_source': 'https://arxiv.org/abs/1606.04797v1', 'reviewed_PDF_pages': [1, 6],
         'mechanism': 'The original foreground soft-overlap objective uses squared predicted/ground-truth terms in its Dice denominator for volumetric medical segmentation.',
         'implemented_use': 'The existing crack loss instead uses linear probability/indicator sums, additive1 smoothing and an image-wise average, combined with image BCE and weighted pixel CE.',
         'code': ['elresearch/full_models.py'],
         'limit': 'This is attribution of an overlap-loss antecedent, not the exact V-Net objective, its architecture or any claim to invent Dice training.'},
    ]
    errors = []
    screened = []
    for row in records:
        key = row['key']
        metadata = json.loads((REFS / (key + '_metadata.json')).read_text())
        errors += [{'key': key, **error} for error in metadata['errors']]
        # A search response is only discovery. Never select a paper from rank alone.
        for hit in metadata['records']:
            selected = ((key == 'adamw' and '1711.05101' in hit.get('url', '')) or
                        (key == 'cutmix' and '1905.04899' in hit.get('url', '')) or
                        (key == 'vnet' and str(hit.get('doi', '')).lower() == '10.1109/3dv.2016.79'))
            screened.append({'query_key': key, 'title': hit['title'], 'url': hit.get('url'),
                             'selected_as_discovery_match': selected,
                             'decision': 'Exact original-method identity; independently read primary PDF' if selected else
                             'Excluded from this targeted review: not the specified original paper'})
        row['evidence_access'] = 'Selected original full-text sections actually read; PDF bytes and extracted text retained locally'
        row['primary_PDF'] = ref(REFS / (key + '.pdf'))
        row['extracted_text'] = ref(REFS / (key + '.txt'))
        row['implementation_bindings'] = [ref(ROOT / path) for path in row['code']]
        row['original_paper_performance_used_as_our_result'] = False
    methods = r'''AdamW~\cite{adamw} separates weight decay from adaptive gradient updates. Our image-only and joint dense systems use the registered AdamW configuration; the full published-architecture port uses SGD with momentum. This is an optimization difference, not evidence of equal training cost or a backbone-specific cause.

Overlap-based training objectives have precedents including V-Net~\cite{vnet}. Our crack-class soft Dice term uses linear probability/indicator sums, additive-one smoothing and a per-image mean. V-Net uses squared denominator terms in its volumetric objective. The implemented loss combination is therefore specified here rather than presented as an exact V-Net reproduction.

Weight-averaged teacher targets, adversarial prediction smoothness and rectangular image mixing build on Mean Teacher~\cite{meanteacher}, VAT~\cite{vat} and CutMix~\cite{cutmix}, respectively. Our port applies VAT in encoder-feature space through two frozen teacher decoders, and mixes spatial pseudo-logits and validity masks with image regions. Its Beta(4,4) setting differs from the original CutMix experiments' Beta(1,1). These are documented adaptations of component principles; original classification or medical-segmentation scores are not matched EL results.
'''
    bibliography = r'''\bibitem{adamw} I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. \emph{ICLR},2019. \url{https://arxiv.org/abs/1711.05101v3}.
\bibitem{meanteacher} A. Tarvainen and H. Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. \emph{Advances in Neural Information Processing Systems},30,2017. \url{https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html}.
\bibitem{cutmix} S. Yun et al. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. \emph{ICCV},2019. \url{https://arxiv.org/abs/1905.04899v2}.
\bibitem{vat} T. Miyato,S.-i. Maeda,M. Koyama,and S. Ishii. Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning. Author manuscript,\emph{arXiv:1704.03976v2},2018 revision. \url{https://arxiv.org/abs/1704.03976v2}.
\bibitem{vnet} F. Milletari,N. Navab,and S.-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. \emph{3DV},2016. \url{https://doi.org/10.1109/3DV.2016.79}.
'''
    (OUT / 'methods_additions.tex').write_text(methods)
    (OUT / 'bibliography_additions.tex').write_text(bibliography)
    save(OUT / 'search_screening.json', {'at': datetime.now(timezone.utc).isoformat(),
         'hits': screened, 'metadata_errors_retained': errors,
         'discovery_misses': ['Mean Teacher and VAT were not returned as exact originals by the bounded helper queries. Primary publisher/arXiv pages and their original PDFs were inspected independently.'],
         'exhaustive_review': False})
    source_paths = [Path(__file__).absolute(), *sorted(REFS.iterdir()), OUT / 'methods_additions.tex',
                    OUT / 'bibliography_additions.tex', OUT / 'search_screening.json']
    code_paths = sorted({path for row in records for path in row['code']})
    source_paths += [ROOT / path for path in code_paths]
    review = {'at': datetime.now(timezone.utc).isoformat(), 'records': records,
              'question': json.loads((REFS / 'review_scope.json').read_text())['question'],
              'fulltext_works': 5, 'systematic_exhaustive_review': False,
              'input_bindings': [ref(path) for path in source_paths],
              'prior_completed_reviews_refetched_or_overwritten': False,
              'scientific_code_protocol_checkpoint_changed': False,
              'new_model_fit_inference_or_bootstrap': False, 'independent_confirmation': False,
              'editorial_correction': 'Describe active-teacher alternation by round separately from EMA after each optimizer update. This corrects prose ambiguity, not training code.',
              'scope': 'Primary method attribution plus exact local implementation distinctions only. Five publication records are not new model experiments, a causal mechanism test, independent superiority or acceptance.'}
    save(OUT / 'review.json', review)
    content = ['<h1>Attribution of implemented training components</h1>',
               '<p>Five selected primary-method full texts; unchanged experiments. Metadata discovery misses and access errors are retained.</p>']
    for row in records:
        content += ['<h2>' + html.escape(row['citation']) + '</h2>',
                    '<p><a href="' + html.escape(row['primary_source']) + '">Primary source</a>; selected PDF pages ' + str(row['reviewed_PDF_pages']) + '.</p>']
        for key in ['mechanism', 'implemented_use', 'limit']:
            content.append('<p><strong>' + key + ':</strong> ' + html.escape(row[key]) + '</p>')
    (OUT / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Training component provenance</title><style>body{font:16px Arial;max-width:1000px;margin:40px auto;line-height:1.5}</style>' + ''.join(content) + '</html>')
    save(OUT / 'outputs.json', {'files': [ref(OUT / name) for name in [
        'review.json', 'search_screening.json', 'methods_additions.tex', 'bibliography_additions.tex', 'report.html']]})
    print({'review': ref(OUT / 'review.json'), 'fulltexts_read': 5, 'scientific_changes': False})


if __name__ == '__main__':
    main()
