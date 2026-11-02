"""Presentation only: cover every frozen contrast, without fitting or resampling.

--prepare reads declarations only. --render additionally requires the completed
original analysis and a successful full reporting review. Display order is the
frozen declaration order, never effect size, interval sign, or apparent merit.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import textwrap

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
FINAL = BASE / 'final_confirmation_v1'
REGISTRY_SHA = 'eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
CORE_QUESTION = 'Original dense adaptation and same-mask CNN/frozen/full/reference controls'
ROWS_PER_PAGE = 18
COHORTS = {
    'scdd_publication_v1': 'SCDD117',
    'ucf_reported_real_confirmation_v1': 'UCF507',
    'pvelad_restricted_confirmation_v2': 'PVEL primary12292 (scored12326)',
}
ENDPOINTS = {
    'AUROC': 'Image AUROC', 'average_precision': 'Image AP',
    'recall_at_source_threshold': 'Source-threshold recall',
    'review_rate_at_source_threshold': 'Source-threshold review rate',
    'false_positive_rate_at_source_threshold': 'Source-threshold image FPR',
    'positive_image_Dice': 'Positive-image Dice',
    'positive_image_IoU': 'Positive-image IoU',
    'normal_structure_pixel_FPR': 'Normal-structure pixel FPR',
    'pooled_native_pixel_AP': 'Whole-cohort pooled pixel AP',
    'known_crack_recall': 'Known-crack recall',
    'author_normal_FPR': 'Author-normal FPR',
    'box_point_hit_rate': 'Crack-box point hit rate',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(p):
    p = Path(p)
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def read(p, expected=None):
    if expected:
        assert sha(p) == expected, str(p)
    return json.loads(Path(p).read_text())


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_once(p, value):
    if p.exists():
        assert read(p) == value, str(p)
    else:
        save(p, value)


def csv_once(p, rows):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    value = stream.getvalue()
    if p.exists():
        assert p.read_bytes() == value.encode(), str(p)
    else:
        p.write_bytes(value.encode())


def family_label(model, family_code):
    pieces = model['family'].split('__')
    source, scenario = pieces[:2]
    return (f"{family_code} | {source.upper()}/{scenario} | "
            f"{100*model['training_fraction']:g}% | "
            f"{pieces[3]} | {model['source_view']}")


def prepare():
    registry_path = FINAL / 'frozen_registry.json'
    registry = read(registry_path, REGISTRY_SHA)
    for b in registry['evaluator_files']:
        assert sha(ROOT / b['path']) == b['sha256'], b['path']
    rationale_path = BASE / 'final_source_review_v1/concrete_specification_v1/contrast_rationale.json'
    bound = next(x for x in registry['analysis_plan']['source_review_inputs']
                 if x['path'] == str(rationale_path.relative_to(ROOT)))
    rationales = {x['id']: x for x in read(rationale_path, bound['sha256'])}
    declarations = registry['analysis_plan']['primary_comparisons']
    assert len(declarations) == len(rationales) == 6212
    models = {m['id']: m for m in registry['models']}
    families = {}
    for m in models.values():
        if m['family'] not in families:
            code = f'F{len(families)+1:03d}'
            families[m['family']] = {'code': code, 'model': m,
                                    'label': family_label(m, code)}
    assert len(families) == 427
    groups = defaultdict(list)
    core_groups = defaultdict(list)
    for c in declarations:
        groups[c['cohort'], c['analysis_stratum'], c['endpoint']].append(c['id'])
        if CORE_QUESTION in rationales[c['id']]['purposes']:
            core_groups[c['cohort'], c['analysis_stratum']].append(c)
    pages = []
    for (cohort, stratum, endpoint), ids in groups.items():
        for start in range(0, len(ids), ROWS_PER_PAGE):
            pages.append({'page': len(pages)+1, 'cohort': cohort, 'stratum': stratum,
                          'endpoint': endpoint, 'contrast_ids': ids[start:start+ROWS_PER_PAGE]})
    included = [x for page in pages for x in page['contrast_ids']]
    assert len(included) == len(set(included)) == 6212
    assert set(included) == {c['id'] for c in declarations}
    core = []
    for (cohort, stratum), cc in core_groups.items():
        pairs = list(dict.fromkeys((c['first_family'], c['second_family']) for c in cc))
        endpoints = list(dict.fromkeys(c['endpoint'] for c in cc))
        core.append({'cohort': cohort, 'stratum': stratum,
                     'family_pairs': [list(x) for x in pairs], 'endpoints': endpoints,
                     'contrast_ids': [c['id'] for c in cc]})
    plan = {'version': 1, 'registry': binding(registry_path),
            'registry_id': registry['registry_id'], 'renderer': binding(Path(__file__)),
            'rationale': bound, 'all_declared_contrasts': 6212, 'all_families': 427,
            'order': 'Frozen cohort/stratum/endpoint and declaration order; no outcome sorting.',
            'atlas_pages': pages, 'core_dense_question_matrices': core,
            'family_codes': {name: {'code': f['code'], 'label': f['label']}
                             for name, f in families.items()},
            'display': {'difference': '100 x (SECOND family - FIRST family)',
                        'forest_axis': [-100, 100], 'matrix_color_scale': [-100, 100],
                        'intervals': 'Original conditional acquisition-family95% intervals, UNADJUSTED.',
                        'seeds': 'Every original fitted pair shown; deterministic reuse is not new training.',
                        'missing': 'Not estimable is explicit, not zero. Not declared is a distinct matrix cell.',
                        'rates': 'Lower FPR/review rate is not automatically better without recall.',
                        'PVEL': 'No capture-family intervals; boxes do not support mask Dice.',
                        'pooled_AP': 'Whole-cohort point estimate only; no pixel pseudoreplication.'},
            'no_new_inference_training_resampling_or_scientific_choice': True,
            'preparation_does_not_read_target_metrics': True,
            'manuscript_or_submission_ready': False}
    out = BASE / 'final_contrast_atlas_v1' / f'registry_{REGISTRY_SHA[:12]}_renderer_{sha(__file__)[:12]}'
    out.mkdir(parents=True, exist_ok=True)
    save_once(out / 'presentation_plan.json', plan)
    csv_once(out / 'family_codes.csv',
             [dict(family=name, **f) for name, f in plan['family_codes'].items()])
    csv_once(out / 'all6212_contrast_page_index.csv',
             [dict(contrast_id=x, page=p['page'], cohort=p['cohort'],
                   stratum=p['stratum'], endpoint=p['endpoint'])
              for p in pages for x in p['contrast_ids']])
    if not (out / Path(__file__).name).exists():
        shutil.copy2(__file__, out / Path(__file__).name)
    assert sha(out / Path(__file__).name) == sha(__file__)
    return registry, declarations, families, plan, out


def draw_forest(page, rows, families, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = len(rows)
    fig = plt.figure(figsize=(18.5, max(7.5, 3.2 + .62*n)))
    ax = fig.add_axes([.54, .19, .26, .66])
    ax.set_xlim(-100, 100)
    ax.set_ylim(n-.5, -.5)
    ax.axvline(0, color='#8695a4', lw=.8)
    ax.set_yticks([])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.tick_params(labelsize=9)
    ax.set_xlabel('100 x (SECOND - FIRST)', fontsize=10)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    fig.text(.025, .956, ENDPOINTS[page['endpoint']], fontsize=18, weight='bold')
    fig.text(.025, .924, f"{COHORTS[page['cohort']]} | Primary stratum: {page['stratum']} | Atlas page {page['page']}", fontsize=11)
    fig.text(.025, .891, 'Frozen declaration order. Symbols show literal signed differences; their colors do not identify a superior system.', fontsize=10, color='#33465b')
    for x, title in [(.844, 'FIRST'), (.904, 'SECOND'), (.964, 'DELTA')]:
        fig.text(x, .868, title, ha='center', fontsize=9, weight='bold')
    for i, row in enumerate(rows):
        c, est = row['declaration'], row['estimate']
        y = .19 + .66*(1-(i+.5)/n)
        first = families[c['first_family']]
        second = families[c['second_family']]
        label = ('FIRST: '+first['label']+'\nSECOND: '+second['label'])
        # Wrap long identifiers while preserving distinct family codes.
        label = '\n'.join(textwrap.fill(s, width=105) for s in label.splitlines())
        fig.text(.025, y, label, va='center', fontsize=7.8, linespacing=1.25)
        if i % 2 == 0:
            ax.axhspan(i-.48, i+.48, color='#f2f5f8', zorder=0)
        if row['status'] == 'not_estimable':
            ax.text(0, i, 'NOT ESTIMABLE', ha='center', va='center', fontsize=8, color='#667588')
            for x in [.844, .904, .964]:
                fig.text(x, y, 'NA', ha='center', va='center', fontsize=9, color='#667588')
            continue
        ci = est['acquisition_family_percentile_ci95']
        if ci is not None:
            ax.plot([100*ci[0], 100*ci[1]], [i, i], color='#172b40', lw=1.6, zorder=2)
        pairs = est['fitted_pairs']
        for j, pair in enumerate(pairs):
            offset = (j-(len(pairs)-1)/2)*.19
            ax.scatter([100*pair['difference']], [i+offset], marker=['o', '^', 's'][j],
                       s=17, color='#8192a3', alpha=.9, zorder=3)
        ax.scatter([100*est['mean_pair_difference']], [i], marker='D', s=22,
                   color='#0a6888', edgecolor='white', linewidth=.35, zorder=4)
        for x, value in [(.844, est['first_mean']), (.904, est['second_mean']), (.964, est['mean_pair_difference'])]:
            fig.text(x, y, f'{100*value:+.2f}' if x == .964 else f'{100*value:.2f}',
                     ha='center', va='center', fontsize=9)
    handles = [Line2D([], [], color='#172b40', label='Original conditional 95% interval (unadjusted)'),
               Line2D([], [], marker='D', color='white', markerfacecolor='#0a6888', label='Mean of fitted-pair differences'),
               Line2D([], [], marker='o', color='white', markerfacecolor='#8192a3', label='Every original fitted pair')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.02, .104), ncol=3, frameon=False, fontsize=9)
    footer = [
        'Values and differences are multiplied by 100 for display only. Rate differences are percentage points; AUROC/AP/Dice/IoU remain score differences.',
        'Intervals condition on these fitted systems and have no multiplicity adjustment. They do not represent uncertainty over new training runs or independent domains.',
        'PVEL and whole-cohort pooled pixel AP have point estimates only. NA retains unsupported endpoints. A lower review/FPR rate must be considered together with recall.',
        'All 6,212 frozen declarations appear once in this atlas. Correlated teacher/ensemble views and deterministic reuse remain disclosed in the companion ledgers.',
    ]
    fig.text(.025, .035, '\n'.join(footer), fontsize=8, color='#33465b', linespacing=1.55)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(destination.with_suffix('.'+extension), dpi=150, facecolor='white')
    plt.close(fig)


def draw_matrix(spec, rows_by_id, families, destination):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    pairs, endpoints = spec['family_pairs'], spec['endpoints']
    lookup = {(v['declaration']['first_family'], v['declaration']['second_family'], v['declaration']['endpoint']): v
              for v in [rows_by_id[x] for x in spec['contrast_ids']]}
    values = np.full((len(pairs), len(endpoints)), np.nan)
    fig, ax = plt.subplots(figsize=(max(13, 1.25*len(endpoints)+6), max(6.4, .64*len(pairs)+3.8)))
    for i, (first, second) in enumerate(pairs):
        for j, endpoint in enumerate(endpoints):
            row = lookup.get((first, second, endpoint))
            if row and row['status'] == 'estimated':
                values[i, j] = 100*row['estimate']['mean_pair_difference']
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#e7ecf1')
    im = ax.imshow(values, cmap=cmap, vmin=-100, vmax=100, aspect='auto', interpolation='none')
    for i, (first, second) in enumerate(pairs):
        for j, endpoint in enumerate(endpoints):
            row = lookup.get((first, second, endpoint))
            if row is None:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch='///', edgecolor='#bac4cf', lw=0))
                label = 'ND'
            elif row['status'] != 'estimated':
                label = 'NE'
            else:
                est = row['estimate']
                signs = ','.join('+' if p['difference'] > 0 else '-' if p['difference'] < 0 else '0' for p in est['fitted_pairs'])
                label = f"{100*est['mean_pair_difference']:+.1f}\n({signs})"
            ax.text(j, i, label, ha='center', va='center', fontsize=8,
                    color='white' if np.isfinite(values[i,j]) and abs(values[i,j]) > 60 else '#172b40')
    ax.set_yticks(range(len(pairs)), [f"{families[a]['code']} -> {families[b]['code']}" for a,b in pairs], fontsize=9)
    ax.set_xticks(range(len(endpoints)), ['\n'.join(textwrap.wrap(ENDPOINTS[e], 17)) for e in endpoints], fontsize=8)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=9)
    ax.set_xticks([i-.5 for i in range(len(endpoints)+1)], minor=True)
    ax.set_yticks([i-.5 for i in range(len(pairs)+1)], minor=True)
    ax.grid(which='minor', color='white', lw=1)
    ax.tick_params(which='minor', length=0)
    fig.suptitle(f"Dense adaptation controls | {COHORTS[spec['cohort']]} | {spec['stratum']}", fontsize=16, weight='bold', y=.985)
    fig.colorbar(im, ax=ax, fraction=.025, pad=.025, label='100 x (SECOND - FIRST); signs are literal, not merit')
    caption = ('Every contrast for the original dense-control question is retained. Parentheses show every fitted-pair direction.\n'
               'ND: no such endpoint was declared for that pair. NE: declared but not estimable. Gray cells are never zeros.\n'
               'Intervals remain in the complete forest atlas; no adjusted-significance claim. Recall and review/FPR must be read together.\n'
               'PVEL boxes cannot support mask Dice; pooled native AP is whole-cohort point only. Family codes resolve in the companion ledger.')
    fig.text(.025, .025, caption, fontsize=8, color='#33465b', linespacing=1.5)
    fig.subplots_adjust(left=.15, right=.92, top=.83, bottom=.21)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(destination.with_suffix('.'+extension), dpi=160, facecolor='white')
    plt.close(fig)


def render(registry, declarations, families, plan, out, review_path, limit):
    import matplotlib
    matplotlib.use('Agg')
    analysis_path = FINAL / 'primary_analysis.json'
    analysis = read(analysis_path)
    review = read(review_path)
    assert review['final_analysis_checked'] is True
    assert review['all6212_declarations_retained'] is True
    assert review['original_registry_id'] == registry['registry_id']
    review_bindings = read(review_path.parent / 'input_bindings.json')
    assert review_bindings[str(analysis_path.relative_to(ROOT))] == sha(analysis_path)
    assert [v['declaration'] for v in analysis['primary_comparisons']] == declarations
    rows_by_id = {v['declaration']['id']: v for v in analysis['primary_comparisons']}
    assert len(rows_by_id) == 6212
    input_proof = {'presentation_plan': binding(out/'presentation_plan.json'),
                   'original_analysis': binding(analysis_path), 'complete_reporting_review': binding(review_path)}
    save_once(out/'render_input_bindings.json', input_proof)
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    receipts = []
    requests = [('atlas', p) for p in plan['atlas_pages']] + [('matrix', s) for s in plan['core_dense_question_matrices']]
    for index, (kind, spec) in enumerate(requests):
        destination = figures/f'{index+1:03d}_{kind}'
        receipt_path = destination.with_suffix('.json')
        if not receipt_path.exists():
            if limit is not None and sum(x['newly_rendered'] for x in receipts) >= limit:
                break
            rows = [rows_by_id[x] for x in spec['contrast_ids']]
            if kind == 'atlas':
                draw_forest(spec, rows, families, destination)
            else:
                draw_matrix(spec, rows_by_id, families, destination)
            receipt = {'at': datetime.now(timezone.utc).isoformat(), 'kind': kind, 'specification': spec,
                       'original_analysis': binding(analysis_path),
                       'outputs': [binding(destination.with_suffix('.'+e)) for e in ['png','pdf','svg']],
                       'visual_QA_complete': False, 'manuscript_integrated': False}
            save(receipt_path, receipt)
            newly = True
        else:
            receipt = read(receipt_path)
            assert receipt['specification'] == spec
            assert receipt['original_analysis'] == binding(analysis_path)
            for b in receipt['outputs']:
                assert sha(ROOT/b['path']) == b['sha256']
            newly = False
        receipts.append({'receipt':binding(receipt_path),'newly_rendered':newly})
        if newly:
            print(json.dumps({'rendered':index+1,'planned':len(requests),'kind':kind}), flush=True)
    for b in [input_proof['original_analysis'],input_proof['presentation_plan'],input_proof['complete_reporting_review']]:
        assert sha(ROOT/b['path']) == b['sha256']
    save(out/'render_status.json', {'at':datetime.now(timezone.utc).isoformat(), 'planned_figures':len(requests),
                                  'rendered_figures':len(receipts),'all_rendered':len(receipts)==len(requests),
                                  'receipts':receipts,'visual_QA_complete':False,'submission_ready':False})


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--render', action='store_true')
    parser.add_argument('--verified-review', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    registry, declarations, families, plan, out = prepare()
    if args.render:
        assert args.verified_review and args.verified_review.is_file()
        assert args.limit is None or args.limit > 0
        render(registry,declarations,families,plan,out,args.verified_review,args.limit)
    print(json.dumps({'output':str(out),'stage':'rendered' if args.render else 'declaration_only_prepared',
                      'atlas_pages':len(plan['atlas_pages']),
                      'core_dense_matrices':len(plan['core_dense_question_matrices']),
                      'all6212covered':True,'target_metrics_read':args.render}))


if __name__ == '__main__':
    main()
