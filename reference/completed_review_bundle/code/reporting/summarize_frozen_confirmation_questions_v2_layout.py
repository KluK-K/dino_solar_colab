"""All declared question themes and joint views of EXISTING endpoint contrasts.

No new contrast, combined endpoint, fitted line, correlation test, bootstrap,
threshold, model ranking or subset is introduced. Each joined point refers to
the same original family pair/cohort/stratum and its already declared estimates.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
FINAL = BASE / 'final_confirmation_v1'
REGISTRY_SHA = 'eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')


def table(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def prepare():
    registry_path = FINAL / 'frozen_registry.json'
    assert sha(registry_path) == REGISTRY_SHA
    registry = read(registry_path)
    rationale_path = BASE / 'final_source_review_v1/concrete_specification_v1/contrast_rationale.json'
    rationale_binding = next(x for x in registry['analysis_plan']['source_review_inputs'] if x['path'] == str(rationale_path.relative_to(ROOT)))
    assert sha(rationale_path) == rationale_binding['sha256']
    rationale = read(rationale_path)
    assert len(rationale) == len(registry['analysis_plan']['primary_comparisons']) == 6212
    themes = list(dict.fromkeys(p for row in rationale for p in row['purposes']))
    out = BASE / 'complete_confirmation_question_readout_v1' / sha(Path(__file__))[:20]
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(version=1, registry=ref(registry_path), rationale=ref(rationale_path),
                reporter=ref(Path(__file__).absolute()), all_declared_contrasts=6212,
                all_question_themes=themes, theme_membership_overlaps=True,
                theme_readout='Every question/cohort/stratum/endpoint: estimated/null counts, literal signed mean-difference and fitted-pair direction counts, original unadjusted interval positions. No mean of unrelated model scores or adjusted-significance test.',
                joint_views=[['AUROC', 'positive_image_Dice'],
                             ['recall_at_source_threshold', 'review_rate_at_source_threshold'],
                             ['known_crack_recall', 'author_normal_FPR'],
                             ['known_crack_recall', 'box_point_hit_rate']],
                join='Exact original FIRST/SECOND family, cohort and analysis stratum; no reversed signs or cross-model AUROC pooling.',
                axes=[-1, 1], points='Every eligible original pair; null joint cells remain in tables with a reason.',
                no_new_contrasts_or_scientific_choice=True, target_outcomes_read_during_preparation=False)
    dest = out / 'presentation_plan.json'
    if dest.exists(): assert read(dest) == plan
    else: save(dest, plan)
    snapshot = out / Path(__file__).name
    if not snapshot.exists(): shutil.copy2(__file__, snapshot)
    assert sha(snapshot) == sha(Path(__file__))
    return registry, rationale, out


def sign(v):
    return 'positive' if v > 0 else 'negative' if v < 0 else 'zero'


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--render', action='store_true')
    parser.add_argument('--verified-review', type=Path)
    args = parser.parse_args()
    registry, rationale, out = prepare()
    if args.prepare:
        print(json.dumps({'prepared': str(out), 'question_themes': len(read(out/'presentation_plan.json')['all_question_themes']), 'target_outcomes_read': False})); return
    assert args.verified_review and args.verified_review.is_file()
    review = read(args.verified_review)
    assert review['final_analysis_checked'] and review['all6212_declarations_retained']
    analysis_path = FINAL / 'primary_analysis.json'
    assert read(args.verified_review.parent/'input_bindings.json')[str(analysis_path.relative_to(ROOT))] == sha(analysis_path)
    analysis = read(analysis_path)
    assert analysis['inputs']['registry_id'] == registry['registry_id']
    comparisons = analysis['primary_comparisons']
    assert [c['declaration'] for c in comparisons] == registry['analysis_plan']['primary_comparisons']
    questions = {q['id']: q for q in rationale}
    grouped = defaultdict(list); indexed = defaultdict(dict)
    for row in comparisons:
        d = row['declaration']
        for theme in questions[d['id']]['purposes']:
            grouped[theme, d['cohort'], d['analysis_stratum'], d['endpoint']].append(row)
        key = d['first_family'], d['second_family'], d['cohort'], d['analysis_stratum']
        assert d['endpoint'] not in indexed[key], 'Do not silently discard duplicated original declarations.'
        indexed[key][d['endpoint']] = row
    run = out / ('render_' + datetime.now(timezone.utc).isoformat().replace(':', '-')); run.mkdir()
    save(run/'registration.json', dict(at=datetime.now(timezone.utc).isoformat(),
        plan=ref(out/'presentation_plan.json'), original_analysis=ref(analysis_path), full_review=ref(args.verified_review)))
    summaries = []
    for (theme, cohort, stratum, endpoint), rows in grouped.items():
        counts = Counter()
        for row in rows:
            if row['status'] == 'not_estimable': counts['not_estimable'] += 1; continue
            counts['estimated'] += 1; est = row['estimate']
            counts['literal_mean_' + sign(est['mean_pair_difference'])] += 1
            directions = [sign(p['difference']) for p in est['fitted_pairs']]
            counts['all_fitted_pairs_' + directions[0] if len(set(directions)) == 1 else 'mixed_fitted_pair_directions'] += 1
            ci = est['acquisition_family_percentile_ci95']
            counts['no_interval' if ci is None else 'interval_entirely_positive' if ci[0] > 0 else 'interval_entirely_negative' if ci[1] < 0 else 'interval_contains_zero'] += 1
        names = ['estimated', 'not_estimable', 'literal_mean_positive', 'literal_mean_negative', 'literal_mean_zero',
                 'all_fitted_pairs_positive', 'all_fitted_pairs_negative', 'all_fitted_pairs_zero', 'mixed_fitted_pair_directions',
                 'interval_entirely_positive', 'interval_entirely_negative', 'interval_contains_zero', 'no_interval']
        summaries.append(dict(question=theme, cohort=cohort, stratum=stratum, endpoint=endpoint,
                              declared=len(rows), **{k: counts[k] for k in names}))
    table(run/'all_question_endpoint_direction_and_null_counts.csv', summaries)
    joint = []
    views = read(out/'presentation_plan.json')['joint_views']
    for xname, yname in views:
        for key, endpoints in indexed.items():
            if xname not in endpoints or yname not in endpoints: continue
            a, b = endpoints[xname], endpoints[yname]
            valid = a['status'] == b['status'] == 'estimated'
            joint.append(dict(first_family=key[0], second_family=key[1], cohort=key[2], stratum=key[3],
                x_endpoint=xname, y_endpoint=yname, x_contrast=a['declaration']['id'], y_contrast=b['declaration']['id'],
                status='both_estimable' if valid else 'one_or_both_declared_endpoints_not_estimable',
                x_difference=a['estimate']['mean_pair_difference'] if a['status']=='estimated' else None,
                y_difference=b['estimate']['mean_pair_difference'] if b['status']=='estimated' else None))
    table(run/'every_original_joint_endpoint_pair.csv', joint)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for number, (xname, yname) in enumerate(views, 1):
        records = [r for r in joint if r['x_endpoint'] == xname and r['y_endpoint'] == yname]
        facets = list(dict.fromkeys((r['cohort'], r['stratum']) for r in records))
        assert facets
        ncols = min(3, len(facets)); nrows = (len(facets)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(12,5*ncols), 4.5*nrows+1), squeeze=False)
        for ax, facet in zip(axes.flat, facets):
            rr = [r for r in records if (r['cohort'],r['stratum']) == facet]
            valid = [r for r in rr if r['status']=='both_estimable']
            if len(facets)==1: ax.set_aspect('equal',adjustable='box')
            ax.set_xlim(-1.03,1.03); ax.set_ylim(-1.03,1.03)
            ax.axhline(0,color='#8695a4',lw=.7); ax.axvline(0,color='#8695a4',lw=.7)
            ax.scatter([r['x_difference'] for r in valid],[r['y_difference'] for r in valid],s=15,alpha=.5,color='#0a6888',linewidths=0)
            ax.grid(color='#dce4ec',lw=.5)
            cohort = 'SCDD' if facet[0].startswith('scdd') else 'UCF507' if facet[0].startswith('ucf') else 'PVEL primary12292'
            ax.set_title(f'{cohort} / {facet[1]} | {len(valid)} points; {len(rr)-len(valid)} null joins',fontsize=10)
            ax.set_xlabel(xname.replace('_',' ')+' difference',fontsize=8)
            ax.set_ylabel(yname.replace('_',' ')+' difference',fontsize=8)
            for side in ['top','right']: ax.spines[side].set_visible(False)
        for ax in list(axes.flat)[len(facets):]: ax.set_visible(False)
        fig.suptitle('Existing paired endpoints | literal SECOND minus FIRST',fontsize=16,weight='bold',y=.98)
        fig.text(.04,.025,'Every eligible original family-pair/cohort/stratum is retained, without outcome ordering or a fitted association. Null joins remain in the table.\n'
                 'Points are correlated recipe contrasts, not independent domains or training replications. Original conditional, unadjusted intervals remain in the complete atlas.\n'
                 'A joint direction does not establish universal superiority; image decisions, mask overlap and coarse box truth retain different meanings.',fontsize=8,color='#33465b',linespacing=1.4)
        fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.87,wspace=.37,hspace=.35)
        for ext in ['pdf','png','svg']: fig.savefig(run/f'joint_{number:02d}.{ext}',dpi=160,facecolor='white')
        plt.close(fig)
    save(run/'verification.json', dict(at=datetime.now(timezone.utc).isoformat(), registration=ref(run/'registration.json'),
        all6212_original_contrasts_used=True, original_question_themes=len({q for q,_,_,_ in grouped}),
        question_memberships_overlap=True, question_endpoint_rows=len(summaries), joint_original_endpoint_pairs=len(joint),
        no_new_contrast_calibration_inference_bootstrap_correlation_test_or_ranking=True,
        literal_signed_directions_not_merit_or_adjusted_significance=True,
        outputs=[ref(p) for p in sorted(run.iterdir()) if p.suffix in ['.pdf','.png','.svg','.csv']],
        actual_visual_QA_complete=False, manuscript_integrated=False, submission_ready=False))
    assert sha(analysis_path) == read(run/'registration.json')['original_analysis']['sha256']
    print(json.dumps({'output':str(run),'themes':len({q for q,_,_,_ in grouped}),'existing_joint_pairs':len(joint)}))


if __name__ == '__main__':
    main()
