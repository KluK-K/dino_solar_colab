"""Trace all original question themes to completed, unchanged contrast records.

Descriptive reporting and interpretation only. Signed counts are not wins,
independent observations, adjusted tests, or new scientific endpoints.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
NOTES = [
('Adaptation across original image controls', 'Most signed AUROC differences favor the original second-family image LoRA systems, but adverse and tied conditions remain. This does not establish a universally improved inspection operating point.', 'Frozen features, full fine-tuning and CNN comparators have different pretraining and optimization histories; budgets and acquisition strata overlap.'),
('Frozen DINO and classical features with the same classifier', 'The declared frozen-feature comparisons mostly have positive AUROC differences, with an adverse condition retained. The result concerns complete feature/preprocessing systems.', 'Sharing a classifier does not match representation pretraining, dimensionality, preprocessing or annotation-dependent HPO.'),
('LoRA and classical controls', 'All declared AUROC mean differences are positive in the original orientation. Thresholded review and transferred PVEL behavior require their own records.', 'Neural adaptation and classical tuning have unequal search and optimization budgets; the comparisons do not isolate representation or training cost.'),
('Adaptation relative to frozen/classical predictors', 'All declared AUROC mean differences are positive, conditional on these source-selected systems and fixed cohorts.', 'The controls overlap the other image themes and are not independent confirmations of the same claim.'),
('Frozen representation systems', 'Signed AUROC differences are predominantly positive but include an adverse comparison. Retain each original feature and processing recipe.', 'Pretraining, geometry, dimensionality and pooling are confounded; an encoder architecture cause is not identified.'),
('Processing scale', 'AUROC differences have both signs. Increasing processing scale is not uniformly beneficial across the declared source and target conditions.', 'Resizing changes the complete input processing system; it does not recover missing physical information or establish a resolution mechanism.'),
('CLS and patch pooling', 'Both signed AUROC directions occur; pooling effects depend on processing size and source condition.', 'CLS, patch mean and concatenation differ in feature geometry and dimensionality. No pooling is selected from confirmation outcomes.'),
('Frozen larger encoder', 'The original larger-encoder comparisons have positive AUROC differences, while positive PVEL recall changes accompany positive author-normal FPR changes.', 'Larger models and pretraining differ. Recall alone does not establish improved operating decisions or resource efficiency.'),
('Model scale', 'Both signed AUROC directions occur, although positive means are more frequent. The full original model-scale conditions remain separate.', 'Parameter count and pretraining are not isolated causes; shared fitted systems and cohorts are correlated.'),
('ResNet and DINO with a common classifier', 'All declared AUROC mean differences are positive in the original orientation.', 'ImageNet and DINO pretraining differ. This is a complete frozen-feature comparison, not a randomized architecture comparison.'),
('Dense adaptation and original same-mask controls', 'AUROC differences are mostly positive, whereas native Dice has nearly balanced positive and adverse means. Image ranking and native overlap do not provide interchangeable evidence.', 'Same masks do not match pretraining, losses, optimization or source model selection. Keep all fitted-pair directions and pooled-AP point estimates.'),
('Normal reference relative to frozen dense prediction', 'AUROC and Dice contain positive and adverse conditions; positive native-AP means do not imply uniform positive-image overlap or review improvements.', 'Memory coverage, retrieval, decoder fitting and source annotation structure remain alternative explanations; no matching-specific cause is established.'),
('Local patch objective and decoder', 'AUROC and Dice both have mixed signed differences. Changing objective and decoder does not produce an endpoint-uniform gain.', 'These are complete systems with different supervision pathways and decoder structure, despite matching original annotation IDs where verified.'),
('Adapted multidepth and local patch systems', 'The frozen declaration is adapted multidepth LoRA FIRST and patch MLP SECOND. All fourteen AUROC differences are negative; eleven of fourteen Dice mean differences are negative.', 'The direction must not be reversed to match the prose label. Encoder adaptation, feature depth and decoder objectives change together.'),
('Source-calibrated normal distance', 'The original normal-distance SECOND comparisons have adverse AUROC and native-AP differences and often larger normal-structure FPR. Their complete source cutoff is retained.', 'A source normal-distance cutoff is not softmax 0.5 or a crack posterior. Matching and supervised decoder changes cannot be separated.'),
('Fresh decoder after encoder adaptation', 'Fresh-head AUROC and Dice means include positive, adverse and tied behavior; extra head optimization can change the complete system.', 'Fresh decoder initialization, encoder freezing and prior adaptation are jointly changed; parent work must be charged separately.'),
('Reference within matched fresh decoders', 'Reference versus no-reference fresh heads have both signed AUROC and Dice differences, including ties. There is no uniform reference benefit.', 'Same trainable count removes one difference but does not identify a unique retrieval mechanism or erase seed sensitivity.'),
('Full post-adaptation system and original parent', 'Both positive and adverse AUROC and Dice effects remain, while native AP includes adverse means. Parent and new decoder stages are distinct fitted components.', 'Additional optimization and changed encoder/decoder roles preclude a reference-only causal interpretation.'),
('Full supervised local published-architecture controls', 'Many original SECOND dense-comparator means improve AUROC and Dice, while native AP retains numerous adverse conditions. The matched UCF LoRA gain transfers with lower mean SCDD Dice.', 'Published-port objectives, pretraining and optimization work differ. These are local ports, not reproductions or rankings of published headline metrics.'),
('Semi-supervised and supervised local ports', 'AUROC has both signed directions and Dice includes adverse and tied means. Unlabelled exposure does not guarantee a native-overlap benefit.', 'The semi-supervised objective and unlabelled images are additional inputs. Teacher views share fitted weights and do not create independent training replications.'),
('Dense controls and semi-supervised local ports', 'All declared AUROC means are positive in the original orientation, but native AP has more adverse than positive means. Dice and operating decisions need their own original records.', 'Source-selected semi-supervised systems, primary teacher and ensemble are correlated; pretraining and optimizer work differ.'),
('Teacher ensemble', 'Original teacher-ensemble differences have both AUROC directions and positive, adverse or tied Dice. Ensembling is not uniformly beneficial.', 'The views share training. Their differences describe inference choices frozen before scoring, not independent fitted-model replications.'),
('Geometry and local evidence aggregation', 'Position and top-5%-versus-mean controls have mixed AUROC and Dice differences. No geometry or pooling setting is selected after scoring.', 'The frozen controls differ in explicit geometry or aggregation; small cohorts and fitted-seed sensitivity limit mechanistic inference.'),
('Stronger image CNN optimization controls', 'All twelve original LoRA-minus-additional-CNN AUROC means and all thirty-six fitted-pair directions are positive; the strongest ResNet-50 SCDD interval still includes zero.', 'The CNN rates were post-hoc source sensitivities but fixed before confirmation. These reused fits and two cohorts are not twelve independent studies or an architecture effect.'),
('Literal numerical port sensitivity', 'All four original SECOND literal-recipe AUROC, Dice and native-AP mean differences are negative. The selected-port and literal-port failures remain visible.', 'The complete numerical recipe changes more than xi; this is not an xi-only causal test. No target-driven repair is made.'),
('Literal-port correlated ensemble', 'The two AUROC differences have opposite directions, while both Dice means increase.', 'The ensemble and teacher share fitting and the literal numerical recipe. Positive Dice does not imply a general ensemble benefit.'),
('Matched UCF20% DINO controls', 'Frozen-to-LoRA and reference-to-LoRA AUROC and Dice means are positive in both SCDD117 and UCF507. These four DINO contrasts do not denote the supervised-port comparison.', 'The 86 training and 144 validation masks are nested within 574 development/HPO images; matched labels do not match pretraining, HPO or optimization.'),
('Matched UCF20% normal reference', 'Both original frozen-to-reference AUROC and Dice mean differences are negative on SCDD117 and UCF507.', 'This is evidence against a universal reference benefit in these fits, not proof that reference information cannot help under other fixed systems.'),
('Original UCF-only reference transfer', 'AUROC, Dice and native AP each retain positive and adverse future-transfer conditions.', 'Only source-fitting exposure differs from target sample identity; both datasets informed study-wide design, and transfer remains conditional on source-selected systems.'),
]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
def save(p, d): p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def tex(s):
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]: s=s.replace(a,b)
    return s

def main():
    reg=BASE/'final_confirmation_v1/frozen_registry.json'
    primary=BASE/'final_confirmation_v1/primary_analysis.json'
    rationale=BASE/'final_source_review_v1/concrete_specification_v1/contrast_rationale.json'
    assert sha(reg)=='eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
    assert sha(primary)=='382f595f14a6be6213dd8b0c65b6df4e4edf7d40725ba00afafb48ab496326a9'
    r=json.loads(reg.read_text()); a=json.loads(primary.read_text()); q=json.loads(rationale.read_text())
    assert [x['declaration'] for x in a['primary_comparisons']]==r['analysis_plan']['primary_comparisons']
    assert len(q)==len(a['primary_comparisons'])==6212
    by={x['declaration']['id']:x for x in a['primary_comparisons']}
    themes=list(dict.fromkeys(t for row in q for t in row['purposes']))
    assert len(themes)==len(NOTES)==29
    at=datetime.now(timezone.utc).isoformat(); out=BASE/'complete_question_interpretation_v1'/at.replace(':','-');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    reviews=[]; rows=[]; fragments=[]
    for index,(theme,(title,finding,limits)) in enumerate(zip(themes,NOTES),1):
        ids=[v['id'] for v in q if theme in v['purposes']]
        groups=defaultdict(list)
        for identifier in ids:
            x=by[identifier];d=x['declaration'];e=x.get('estimate',{})
            groups[d['endpoint']].append(x)
            rows.append(dict(theme=index,question=theme,**d,status=x['status'],
                first_mean=e.get('first_mean'),second_mean=e.get('second_mean'),
                difference=e.get('mean_pair_difference'),
                fitted_pair_differences=json.dumps([p['difference'] for p in e.get('fitted_pairs',[])])))
        endpoint_summary={}
        for endpoint, xs in groups.items():
            counter=Counter('not_estimable' if x['status']!='estimated' else
                ('positive' if x['estimate']['mean_pair_difference']>0 else 'negative' if x['estimate']['mean_pair_difference']<0 else 'zero') for x in xs)
            endpoint_summary[endpoint]=dict(declared=len(xs),**{k:counter[k] for k in ['positive','negative','zero','not_estimable']})
        reviews.append(dict(index=index,original_question=theme,title=title,original_contrast_ids=ids,
            all_original_family_orientations_retained=True,endpoint_literal_mean_direction_counts=endpoint_summary,
            evidence_interpretation=finding,alternative_explanations_and_limits=limits,
            selection_or_new_scientific_test=False))
        fragments.append(r'\subsection{Q'+str(index)+': '+tex(title)+'}\n'+tex(finding)+' '+tex(limits)+'\n')
    assert set(x['id'] for x in rows)==set(by)
    with (out/'all_question_contrast_interpretation_trace.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'all29_question_interpretations.tex').write_text('\n'.join(fragments))
    save(out/'review.json',dict(at=at,original_primary=ref(primary),registry=ref(reg),rationale=ref(rationale),
        themes=reviews,all6212_contrasts_retained=True,all29_themes_reviewed=True,
        direction='Original SECOND minus FIRST; positive is not automatically favorable.',
        counts_not_independent_wins_or_hypothesis_tests=True,theme_membership_overlaps=True,
        intervals_conditional_unadjusted=True,no_new_models_endpoints_thresholds_populations_or_examples=True))
    save(out/'verification.json',dict(at=at,original_primary=ref(primary),registry=ref(reg),
        all6212_unique_contrasts=len(by),overlapping_question_memberships=len(rows),question_themes=29,
        review=ref(out/'review.json'),trace=ref(out/'all_question_contrast_interpretation_trace.csv'),
        article_fragment=ref(out/'all29_question_interpretations.tex'),scientific_choices_unchanged=True,
        manuscript_integration_pending=True))
    print(json.dumps(dict(folder=str(out),themes=29,unique_contrasts=len(by),memberships=len(rows))))

if __name__=='__main__':main()
