"""Read-only candidate identity audit before a NEW final scientific selection.

This is not preparation/freezing of a registry and never constructs a scorer,
loads weights into a model, reads target pixels, or evaluates predictions.
All original binding routines are called unchanged. Candidate eligibility is
technical only: the later source review must still declare the scientific scope.
"""
import collections
import os
from pathlib import Path
import traceback

from elresearch.common import ROOT, digest, now, read_json, save_json, sha256
from elresearch import confirmation_registry as gate

BASE = ROOT / 'artifacts/publication_20261017'
CATALOG = BASE / 'confirmation_preparation_v1/complete_source_candidate_catalog_20261027_v1.json'
OUT = BASE / 'final_source_review_v1/candidate_identity_audit'


def ref(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def main():
    if OUT.exists():
        raise FileExistsError('Retain this audit and use a registered new version for a repair')
    if (BASE / 'final_confirmation_v1/frozen_registry.json').exists() or (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('This pre-observation audit cannot change an existing final decision')
    OUT.mkdir(parents=True)
    guard_paths = [BASE/'confirmation_preparation_v1/candidate_catalog.json',
                   BASE/'confirmation_preparation_v1/unreviewed_selection.json',
                   BASE/'confirmation_preparation_v1/cohort_analysis_policy_v1.json',
                   BASE/'confirmation_preparation_v1/secondary_native_failure_policy_v1.json']
    guards = [ref(p) for p in guard_paths]
    candidates = read_json(CATALOG)['candidates']
    assert len(candidates) == len({c['id'] for c in candidates})
    source_binding = ref(BASE/'source_development_identity_v1.json')
    source = read_json(ROOT/source_binding['path'])
    assert digest({k:v for k,v in source.items() if k!='signature'}) == source['signature']
    for b in source['source_manifests']:
        gate.check_file(b)
    source_rows = {r['id']:r for r in source['records']}
    targets = {}
    cohort_bindings = []
    for name in sorted(gate.ALLOWED_COHORTS):
        b = gate.cohort_binding(name)
        cohort_bindings.append(b)
        rows = [r for r in read_json(gate.check_file(b['manifest']))['records'] if r['partition']==b['partition']]
        targets[name] = {'ids':{r['id'] for r in rows},
                         'orbits':{r['d4_pixel_sha256'] for r in rows},
                         'groups':{(r['dataset'],r['split_group']) for r in rows if r.get('split_group') is not None}}
    registration = {'at':now(), 'code':ref(Path(__file__)), 'catalog':ref(CATALOG),
                    'source_identity_index':source_binding, 'cohorts':cohort_bindings,
                    'original_files_to_preserve':guards, 'candidates':len(candidates),
                    'original_binding_code':ref(ROOT/'elresearch/confirmation_registry.py'),
                    'scope':'Technical identity/exposure/reload-receipt audit only; not scientific selection, preparation, a final registry or new inference.'}
    save_json(OUT/'registration.json', registration)
    checked = []
    families = collections.defaultdict(list)
    try:
        for candidate in candidates:
            d = candidate['declaration']
            run_path = ROOT/d['run_folder']/'run.json'
            assert sha256(run_path) == candidate['run_sha256']
            model = gate.bind_model(d['kind'],d['family'],d['seed'],ROOT/d['run_folder'],ROOT/d['source_protocol'],d['source_view'])
            assert model['id'] == candidate['id']
            assert model['checkpoint']['sha256'] == candidate['checkpoint_declared_sha256']
            known = set(gate.model_source_ids(model))
            assert known <= set(source_rows)
            assert not set(model['source_train_ids']) & set(model['source_validation_ids'])
            src = [source_rows[i] for i in known]
            orbits = {r['d4_pixel_sha256'] for r in src}
            groups = {(r['dataset'],r['split_group']) for r in src}
            overlaps = {}
            for name,t in targets.items():
                counts = {'identity':len(known & t['ids']), 'D4_orbit':len(orbits & t['orbits']),
                          'provided_acquisition_family':len(groups & t['groups'])}
                assert not any(counts.values()), (model['id'],name,counts)
                overlaps[name] = counts
            families[model['family']].append(model)
            # Full private model binding stays local; contains original source IDs.
            binding_path = OUT/'bindings'/(model['id']+'.json')
            binding_path.parent.mkdir(exist_ok=True)
            save_json(binding_path,model)
            checked.append({'id':model['id'],'binding':ref(binding_path),
                            'candidate_run':model['run'],'checkpoint':model['checkpoint'],
                            'source_threshold':model['source_threshold'],'source_temperature':model['source_temperature'],
                            'source_training_count':len(model['source_train_ids']),
                            'source_validation_count':len(model['source_validation_ids']),
                            'unlabelled_source_count':len(model['source_unlabelled_ids']),
                            'union_source_and_selection_count':len(known),
                            'source_and_selection_identity_set_sha256':digest(sorted(known)),
                            'metadata_overlap_counts':overlaps})
            if len(checked)%20 == 0:
                save_json(OUT/'status.json',{'at':now(),'state':'running','pid':os.getpid(),
                                           'checked':len(checked),'planned':len(candidates),'current':model['id']})
                print('checked',len(checked),'of',len(candidates),flush=True)
        seed_coverage = []
        for family,models in sorted(families.items()):
            seeds = sorted(m['seed'] for m in models)
            deterministic = all((m['kind']=='classical' or m.get('architecture')=='dinov3_normal_distance') and m['training_fraction']==1 for m in models)
            assert seeds == ([11] if deterministic else [11,22,33]), (family,seeds)
            seed_coverage.append({'family':family,'seeds':seeds,'deterministic_full_budget':deterministic})
        assert guards == [ref(p) for p in guard_paths]
        assert not (BASE/'final_confirmation_v1/observation_opened.json').exists()
        result = {'at':now(),'registration':ref(OUT/'registration.json'),'checked_candidates':len(checked),
                  'checked_families':len(families),'models':checked,'family_seed_coverage':seed_coverage,
                  'all_original_binding_checks_passed':True,'all_declared_source_identity_D4_family_overlaps_zero':True,
                  'original_files_preserved':guards,'inference_or_weight_load_performed':False,
                  'target_pixels_or_predictions_read':False,'scientific_selection_completed':False,
                  'final_registry_created':False,
                  'limits':['Existing source reload receipts are checked; weights are byte hashed, not re-executed.',
                            'D4/identity/provided-family checks do not establish absence of all crops, near copies or pretraining membership.',
                            'Every candidate passing this audit is not a decision to score every candidate.']}
        save_json(OUT/'results.json',result)
        save_json(OUT/'status.json',{'at':now(),'state':'complete','checked':len(checked),'families':len(families),
                                   'result':ref(OUT/'results.json'),'scientific_selection_completed':False})
        print('COMPLETE',len(checked),'candidates',len(families),'families',flush=True)
    except BaseException as error:
        save_json(OUT/'status.json',{'at':now(),'state':'failed','checked':len(checked),'planned':len(candidates),
                                   'error':str(error),'traceback':traceback.format_exc(),
                                   'original_source_or_evaluator_changed':False})
        raise


if __name__ == '__main__':
    main()
