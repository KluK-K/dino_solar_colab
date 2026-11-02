"""Read-only reporting review. Never runs models, resamples, or changes a registry.

--scope-only inventories existing declarations without reading target metrics.
--verify-analysis checks the completed original analysis, retaining every record.
This reporting review does not replace the frozen scientific workflow.
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
import statistics

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
FINAL = BASE / 'final_confirmation_v1'
REGISTRY_SHA = 'eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
REGISTRY_ID = '868f841774008801a57081a8406b93e476fdd3084ea594c3988e44baaafa1daf'
SUBGROUPS = ['positive_le_0.5pct', 'positive_0.5_to_2pct', 'positive_gt_2pct', 'all_positives', 'all_negatives']
BINDINGS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    digest = sha(path)
    if expected is not None:
        assert digest == expected, str(path)
    BINDINGS[str(path.relative_to(ROOT))] = digest
    return path


def read(path, expected=None):
    return json.loads(bound(path, expected).read_text())


def refread(binding):
    return read(binding['path'], binding['sha256'])


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(path, records):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def finite_tree(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, list):
        for v in value:
            finite_tree(v)


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)


def job_key(j):
    return j['model_id'], j['cohort'], j['analysis_stratum']


def deterministic(m):
    return m['training_fraction'] == 1 and (m['kind'] == 'classical' or m['architecture'] == 'dinov3_normal_distance')


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--scope-only', action='store_true')
    modes.add_argument('--verify-analysis', action='store_true')
    args = parser.parse_args()
    registry = read(FINAL / 'frozen_registry.json', REGISTRY_SHA)
    assert registry['registry_id'] == REGISTRY_ID
    for item in registry['evaluator_files']:
        bound(item['path'], item['sha256'])
    plan = registry['analysis_plan']
    policies = {key: refread(plan[key]) for key in [
        'statistics_plan_binding', 'cohort_analysis_policy_binding', 'secondary_native_failure_policy_binding']}
    for binding in plan['source_review_inputs']:
        bound(binding['path'], binding['sha256'])
    refread(plan['specification'])
    rationale_path = BASE / 'final_source_review_v1/concrete_specification_v1/contrast_rationale.json'
    assert any(r['path'] == str(rationale_path.relative_to(ROOT)) for r in plan['source_review_inputs'])
    rationale = read(rationale_path)
    by_rationale = {x['id']: x for x in rationale}
    declarations = plan['primary_comparisons']
    assert len(declarations) == len(by_rationale) == 6212
    assert {x['id'] for x in declarations} == set(by_rationale)
    models = {m['id']: m for m in registry['models']}
    families = defaultdict(list)
    for m in models.values():
        families[m['family']].append(m)
    assert len(models) == 1061 and len(families) == 427
    assert Counter(len(v) for v in families.values()) == {3: 317, 1: 110}
    for group in families.values():
        assert sorted(m['seed'] for m in group) == ([11, 22, 33] if len(group) == 3 else [11])
        if len(group) == 1:
            assert deterministic(group[0])
    covered = {x[k] for x in declarations for k in ['first_family', 'second_family']}
    assert covered == set(families)
    jobs = plan['confirmation_jobs']
    assert len(jobs) == len({job_key(j) for j in jobs}) == 1683
    status = read(FINAL / 'execution_status.json')
    assert status['state'] == 'complete' and status['completed_jobs'] == 1683
    audit = read(FINAL / 'execution_audit.json')
    assert audit['registry_id'] == REGISTRY_ID and audit['verified_jobs'] == len(audit['jobs']) == 1683
    assert [job_key(j) for j in audit['jobs']] == [job_key(j) for j in jobs]
    cohorts = {c['name']: c for c in registry['cohorts']}
    population = policies['cohort_analysis_policy_binding']
    manifests = {}
    for name, cohort in cohorts.items():
        manifest = refread(cohort['manifest'])
        rows = [row for row in manifest['records'] if row['partition'] == cohort['partition']]
        assert len(rows) == cohort['images']
        manifests[name] = rows
    expected_populations = []
    for cohort, stratum in dict.fromkeys((j['cohort'], j['analysis_stratum']) for j in jobs):
        rows = manifests[cohort]
        excluded = set(population['cohorts'].get(cohort, {}).get('subsets', {}).get('primary', {}).get('excluded_ids', []))
        selected = [r for r in rows if r['id'] not in excluded and (stratum == 'whole' or r['acquisition_source'] == stratum)]
        expected_populations.append(dict(cohort=cohort, stratum=stratum, images=len(selected),
            positive=sum(r['target'] for r in selected), negative=sum(1-r['target'] for r in selected),
            original_scored_pool=len(rows), excluded_members=len(excluded),
            physical_capture_family_verified=False))
    expected = {(x['cohort'], x['stratum']): x for x in expected_populations}
    assert expected['pvelad_restricted_confirmation_v2', 'whole']['images'] == 12292
    declaration_rows = []
    for c in declarations:
        q = by_rationale[c['id']]
        declaration_rows.append(dict(contrast_id=c['id'], first_family=c['first_family'], second_family=c['second_family'],
            cohort=c['cohort'], stratum=c['analysis_stratum'], endpoint=c['endpoint'],
            first_fitted_predictors=len(families[c['first_family']]), second_fitted_predictors=len(families[c['second_family']]),
            question=' | '.join(q['purposes']), source_unit=json.dumps(q['unit']),
            difference_direction=q['difference_direction'], matched_annotation_ids=c['matched_annotation_identities'],
            matched_compute_HPO_objectives_pretraining_implied=False))
    family_rows = [dict(family=name, fitted_records=len(group), seeds=';'.join(str(m['seed']) for m in group),
        kind=group[0]['kind'], architecture=group[0]['architecture'], inference_view=group[0]['source_view'],
        training_fraction=group[0]['training_fraction'], deterministic_singleton=len(group) == 1,
        declared_contrasts=sum(name in [c['first_family'], c['second_family']] for c in declarations))
        for name, group in families.items()]
    report = dict(at=datetime.now(timezone.utc).isoformat(), stage='reporting_scope_verified' if args.scope_only else 'complete_original_analysis_coverage_verified',
        original_registry_id=REGISTRY_ID, declared_jobs=1683, declared_contrasts=6212,
        model_or_inference_records=1061, recipe_families=427, three_seed_families=317, deterministic_singletons=110,
        all_families_have_declared_questions=True, complete_original_execution_audit=1683,
        expected_populations=expected_populations, endpoint_counts=dict(Counter(c['endpoint'] for c in declarations)),
        planned_native_failure_jobs=sum(models[j['model_id']]['kind'] == 'dense' and j['cohort'] in policies['secondary_native_failure_policy_binding']['allowed_native_cohorts'] for j in jobs),
        dense_spatial_jobs=sum(models[j['model_id']]['kind'] == 'dense' for j in jobs),
        PVEL_box_spatial_jobs=sum(models[j['model_id']]['kind'] == 'dense' and j['cohort'].startswith('pvelad') for j in jobs),
        no_new_training_inference_resampling_or_endpoint=True, no_scientific_source_or_choice_changed=True,
        final_analysis_checked=False, manuscript_or_submission_ready=False)
    contrast_rows, seed_rows, subgroup_rows = [], [], []
    if args.verify_analysis:
        result = read(FINAL / 'primary_analysis.json')
        finite_tree(result)
        assert result['inputs']['registry_id'] == REGISTRY_ID
        assert result['inputs']['analysis_code_sha256'] == sha(ROOT / 'elresearch/confirmation_analysis.py')
        assert result['inputs']['execution_audit']['sha256'] == sha(FINAL / 'execution_audit.json')
        assert result['inputs']['statistics_plan'] == plan['statistics_plan_binding']
        assert result['all_frozen_primary_comparisons_reported'] is True
        assert [v['declaration'] for v in result['primary_comparisons']] == declarations
        assert [job_key(j) for j in result['jobs']] == [job_key(j) for j in jobs]
        for actual, job in zip(result['jobs'], jobs):
            model = models[job['model_id']]
            pop = expected[job['cohort'], job['analysis_stratum']]
            assert actual['primary_population']['analyzed_images'] == actual['metrics']['n'] == pop['images']
            assert actual['metrics']['positive'] == pop['positive']
            assert actual['primary_population']['underlying_scored_images'] == pop['original_scored_pool']
            if job['cohort'] in population['cohorts']:
                subsets = population['cohorts'][job['cohort']]['subsets']
                assert set(actual['population_sensitivities']) == set(subsets) - {'primary'}
                for key, sensitivity in actual['population_sensitivities'].items():
                    assert sensitivity['metrics']['n'] == subsets[key]['images']
                    assert sensitivity['metrics']['positive'] == subsets[key]['positive']
            native = actual['secondary_native_failure_analysis']
            native_expected = model['kind'] == 'dense' and job['cohort'] in policies['secondary_native_failure_policy_binding']['allowed_native_cohorts']
            assert (native is not None) == native_expected
            if native is not None:
                assert native['model_seed'] == model['seed'] and native['analyzed_images'] == pop['images']
                assert native['positive_area_bin_edges'] == [0, .005, .02, 1]
                assert native['confidence_interval'] is None
                assert [s['subgroup'] for s in native['subgroups']] == SUBGROUPS
                by_subgroup = {s['subgroup']: s for s in native['subgroups']}
                assert by_subgroup['all_positives']['images'] == pop['positive']
                assert by_subgroup['all_negatives']['images'] == pop['negative']
                assert sum(by_subgroup[s]['images'] for s in SUBGROUPS[:3]) == pop['positive']
                for subgroup in native['subgroups']:
                    if subgroup['images'] == 0:
                        # Empty denominators imply null rates/means. Literal
                        # event counts may correctly be zero, never an estimate.
                        assert all((v is None or v == 0) if k.endswith('_images') else v is None
                                   for k, v in subgroup.items() if k not in ['subgroup', 'images'])
                    subgroup_rows.append(dict(model_id=job['model_id'], family=model['family'], seed=model['seed'],
                        cohort=job['cohort'], stratum=job['analysis_stratum'], **subgroup))
            if job['cohort'] == 'ucf_reported_real_confirmation_v1' and model['kind'] == 'dense':
                assert actual['localization_point_estimates']['normal_structure_pixel_FPR'] is None
        for declaration_row, actual in zip(declaration_rows, result['primary_comparisons']):
            c = actual['declaration']; est = actual['estimate']
            row = {**declaration_row, 'status': actual['status'], 'reason': actual.get('reason'),
                'first_mean': None, 'second_mean': None, 'difference': None, 'ci95_low': None, 'ci95_high': None,
                'interval_status': None, 'fitted_pair_count': None}
            if actual['status'] == 'not_estimable':
                assert est is None and actual['reason'] and actual['substitute_endpoint_selected'] is False
            else:
                assert actual['status'] == 'estimated' and est is not None
                pairs = est['fitted_pairs']; assert pairs
                first = families[c['first_family']]; second = families[c['second_family']]
                assert est['first_unique_fitted_predictors'] == len(first)
                assert est['second_unique_fitted_predictors'] == len(second)
                assert len(pairs) == max(len(first), len(second))
                aa = sorted(first, key=lambda m: m['seed'])
                bb = sorted(second, key=lambda m: m['seed'])
                if len(aa) == len(bb):
                    expected_pairs = [(x['id'], y['id']) for x, y in zip(aa, bb)]
                elif len(aa) == 1:
                    assert deterministic(aa[0])
                    expected_pairs = [(aa[0]['id'], y['id']) for y in bb]
                else:
                    assert len(bb) == 1 and deterministic(bb[0])
                    expected_pairs = [(x['id'], bb[0]['id']) for x in aa]
                assert [(v['first_model_id'], v['second_model_id']) for v in pairs] == expected_pairs
                for pair in pairs:
                    assert pair['first_model_id'] in {m['id'] for m in first}
                    assert pair['second_model_id'] in {m['id'] for m in second}
                    assert pair['first_seed'] == models[pair['first_model_id']]['seed']
                    assert pair['second_seed'] == models[pair['second_model_id']]['seed']
                    assert 0 <= pair['first_value'] <= 1 and 0 <= pair['second_value'] <= 1
                    close(pair['second_value']-pair['first_value'], pair['difference'])
                    seed_rows.append(dict(contrast_id=c['id'], endpoint=c['endpoint'], cohort=c['cohort'], stratum=c['analysis_stratum'], **pair))
                close(statistics.mean(p['difference'] for p in pairs), est['mean_pair_difference'])
                close(statistics.mean(p['first_value'] for p in pairs), est['first_mean'])
                close(statistics.mean(p['second_value'] for p in pairs), est['second_mean'])
                ci = est['acquisition_family_percentile_ci95']
                if c['endpoint'] == 'pooled_native_pixel_AP' or c['cohort'].startswith('pvelad'):
                    assert ci is None
                if ci is not None:
                    assert len(ci) == 2 and ci[0] <= ci[1]
                    assert est['requested_repetitions'] == policies['statistics_plan_binding']['repetitions']
                    assert est['bootstrap_seed'] == policies['statistics_plan_binding']['seed']
                row.update(first_mean=est['first_mean'], second_mean=est['second_mean'], difference=est['mean_pair_difference'],
                    ci95_low=ci[0] if ci else None, ci95_high=ci[1] if ci else None,
                    interval_status=est['interval_status'], fitted_pair_count=len(pairs))
            contrast_rows.append(row)
        report.update(final_analysis_checked=True, all6212_declarations_retained=True,
            contrast_status_counts=dict(Counter(v['status'] for v in contrast_rows)),
            paired_seed_records=len(seed_rows), native_failure_subgroup_records=len(subgroup_rows),
            empty_native_subgroups=sum(r['images'] == 0 for r in subgroup_rows),
            actual_native_failure_jobs=len(subgroup_rows)//5, PVEL_family_intervals_invented=False,
            pooled_native_AP_intervals_invented=False, raw_ratio_units_preserved=True)
    stamp = datetime.now(timezone.utc).isoformat().replace(':', '-')
    out = BASE / 'final_reporting_review_v1' / stamp
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    table(out / 'all6212_declared_questions.csv', declaration_rows)
    table(out / 'all427_recipe_families.csv', family_rows)
    if args.verify_analysis:
        table(out / 'all6212_original_contrast_results.csv', contrast_rows)
        table(out / 'all_original_paired_seed_results.csv', seed_rows)
        table(out / 'all_native_failure_subgroups.csv', subgroup_rows)
    save(out / 'verification.json', report)
    save(out / 'input_bindings.json', BINDINGS)
    for path, digest in BINDINGS.items():
        assert sha(ROOT / path) == digest
    rows_html = ''.join('<tr><td>'+html.escape(str(x['cohort']))+'</td><td>'+html.escape(x['stratum'])+'</td><td>'+str(x['images'])+'</td><td>'+str(x['positive'])+'</td></tr>' for x in expected_populations)
    state = 'Complete original analysis checked' if args.verify_analysis else 'Frozen reporting scope checked; statistical outcomes still pending'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen confirmation reporting review</title><style>body{font:17px system-ui;color:#172b40;max-width:1100px;margin:48px auto;padding:24px;line-height:1.6}td,th{padding:9px 16px;border-bottom:1px solid #ddd;text-align:left}</style>'
    document += '<h1>'+state+'</h1><p>1061 model/inference records; 427 recipe families; 1683 completed and originally audited jobs; 6212 declared endpoint comparisons. This is a reporting review, not a new scientific specification, analysis, model selection or readiness claim.</p>'
    document += '<table><tr><th>Cohort</th><th>Stratum</th><th>Images analyzed per job</th><th>Crack-positive</th></tr>'+rows_html+'</table>'
    document += '<p>Every recipe family has a declared control question. Source-held systems use their matching SCDD stratum. Deterministic predictor reuse and correlated teacher/ensemble views are not new training replication. All 6212 questions and 427 families are retained in the companion CSVs.</p><p>Intervals condition on fitted systems and remain unadjusted. PVEL has no verified capture-family interval; box truth does not support mask Dice. Pooled native AP is point-only. Review workload must be interpreted with recall; image discrimination and crack overlap remain separate. UCF/SCDD study-wide design exposure remains disclosed.</p></html>'
    (out / 'report.html').write_text(document)
    save(out / 'file_manifest.json', {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps({'output': str(out), 'stage': report['stage'], 'final_analysis_checked': report['final_analysis_checked'], 'jobs':1683, 'comparisons':6212, 'families':427}))


if __name__ == '__main__':
    main()
