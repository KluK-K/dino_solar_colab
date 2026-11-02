"""Amend only seed-family grouping in the preserved unreviewed v1 inventory.

Fitted cutoffs/parent weights/archive heads remain bound to each original run.
They are not architecture choices distinguishing three seeds of the same recipe.
No candidate, source value, checkpoint, model file or target decision is changed.
"""
import collections
import copy
from pathlib import Path
import re
import shutil

from elresearch.common import ROOT, read_json, save_json, sha256, digest, now

BASE = ROOT/'artifacts/publication_20261017'
PREP = BASE/'confirmation_preparation_v1'
SOURCE = PREP/'complete_source_candidate_catalog_20261027_v1.json'
DEST = PREP/'complete_source_candidate_catalog_20261027_v2.json'
REPAIR = BASE/'final_source_review_v1/candidate_grouping_amendment_v2'
FITTED_FIELDS = {'pixel_distance_cutoff','adaptation_parent_run','adaptation_parent_run_sha256',
                 'adapted_encoder_path','adapted_weights_sha256'}


def ref(path):
    return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha256(path)}


def recipe_specification(spec):
    spec = copy.deepcopy(spec)
    for key in FITTED_FIELDS:
        spec.pop(key,None)
    if 'archive_basis' in spec:
        # Preserve fixed archive architecture/backbone semantics. Each random
        # head basis, seed and path stays exactly pinned in the model binding.
        basis = spec['archive_basis']
        spec['archive_basis'] = {key:basis[key] for key in ['backbone_sha256','architecture_commit','purpose']}
    return spec


def main():
    if DEST.exists() or REPAIR.exists():
        raise FileExistsError('Preserve existing registration/output; never overwrite a review version')
    audit = BASE/'final_source_review_v1/candidate_identity_audit'
    failure = read_json(audit/'status.json')
    assert failure['state']=='failed' and failure['checked']==1061
    assert 'seed' in failure['traceback'] and 'dinov3_global_normal_distance' in failure['error']
    assert len(list((audit/'bindings').glob('*.json')))==1061
    source = read_json(SOURCE)
    guards = [ref(p) for p in [SOURCE,PREP/'candidate_catalog.json',PREP/'unreviewed_selection.json',
                              ROOT/'build_confirmation_catalog.py',ROOT/'audit_complete_confirmation_candidates_v1.py']]
    REPAIR.mkdir(parents=True)
    for name,path in [('audit_source_v1.py',ROOT/'audit_complete_confirmation_candidates_v1.py'),
                      ('original_inventory_builder.py',ROOT/'build_confirmation_catalog.py'),
                      ('audit_registration_v1.json',audit/'registration.json'),('audit_failure_v1.json',audit/'status.json')]:
        shutil.copy2(path,REPAIR/name)
    save_json(REPAIR/'registration.json',{
        'at':now(),'code':ref(Path(__file__)),'preserved_inputs':guards,
        'cause':'v1 family digest included fitted distance cutoffs, seed-specific post-adaptation identities and exact random-head archive bases; it separated members of one registered three-seed recipe.',
        'repair_scope':'Reporting/inventory grouping only. Original training/evaluator/registry code, protocols, runs, thresholds, predictions and checkpoints are unchanged.',
        'normalization':{'omit_from_recipe_only':sorted(FITTED_FIELDS),
                         'archive_basis_keep':['backbone_sha256','architecture_commit','purpose']},
        'models_running_or_stopped_for_this_repair':0,'target_outcomes_opened':False,
        'old_failure_and_code_archived':True})
    candidates = []
    mapping = []
    recipes = {}
    for old in source['candidates']:
        c = copy.deepcopy(old)
        run = read_json(ROOT/c['declaration']['run_folder']/'run.json')
        assert sha256(ROOT/c['declaration']['run_folder']/'run.json')==c['run_sha256']
        recipe = recipe_specification(run.get('model',run.get('specification',{})))
        method = re.sub(r'[^A-Za-z0-9_-]','_',c['method'])
        identity = {'study':c['declaration']['source_protocol'],'source_dataset':c['source_dataset'],
                    'fold':c['fold'],'budget':c['training_fraction'],'method':c['method'],
                    'recipe_specification':recipe}
        family = f"{c['source_dataset']}__{c['fold']}__b{c['training_fraction']:g}__{method}__{digest(identity)[:12]}".replace('.','p')
        if c['inference_view']!='primary':
            family += '__'+c['inference_view']
        c['declaration']['family'] = family
        c['id'] = f"{family}_seed{c['seed']}"
        mapping.append({'v1_id':old['id'],'v2_id':c['id'],'v1_family':old['declaration']['family'],
                        'v2_family':family,'seed':c['seed'],'run_sha256':c['run_sha256'],
                        'checkpoint_sha256':c['checkpoint_declared_sha256']})
        unchanged_old = copy.deepcopy(old)
        unchanged_new = copy.deepcopy(c)
        for value in [unchanged_old,unchanged_new]:
            value.pop('id');value['declaration'].pop('family')
        assert unchanged_old==unchanged_new
        recipes.setdefault(family,{'identity':identity,'seeds':[]})['seeds'].append(c['seed'])
        candidates.append(c)
    assert len(candidates)==1061 and len({c['id'] for c in candidates})==1061
    coverage = collections.Counter(tuple(sorted(r['seeds'])) for r in recipes.values())
    assert set(coverage) == {(11,),(11,22,33)}
    assert len(recipes)==427 and coverage[(11,)]==110 and coverage[(11,22,33)]==317
    result = {k:v for k,v in source.items() if k not in {'at','candidates'}}
    result.update({'at':now(),'candidates':sorted(candidates,key=lambda c:c['id']),
                   'grouping_amendment':ref(REPAIR/'registration.json'),
                   'preserved_v1_catalog':ref(SOURCE),
                   'kind':'Complete unreviewed source inventory with seed-invariant recipe-family identities. No selection, registry or scoring.'})
    save_json(DEST,result)
    save_json(REPAIR/'candidate_id_mapping.json',mapping)
    save_json(REPAIR/'family_recipes.json',recipes)
    assert guards == [ref(ROOT/g['path']) for g in guards]
    save_json(PREP/'complete_source_candidate_catalog_20261027_v2_verification.json',{
        'at':now(),'code':ref(Path(__file__)),'inventory':ref(DEST),'amendment':ref(REPAIR/'registration.json'),
        'mapping':ref(REPAIR/'candidate_id_mapping.json'),'family_recipes':ref(REPAIR/'family_recipes.json'),
        'preserved_inputs':guards,'candidates':1061,'families_v1':527,'families_v2':427,
        'three_seed_families':317,'deterministic_singleton_families':110,
        'all_candidate_fields_except_id_and_family_exactly_unchanged':True,
        'all_source_fits_checkpoints_thresholds_scores_unchanged':True,
        'final_registry_created':False,'target_outcomes_opened':False,
        'new_final_source_review_complete':False})
    print('v2 inventory:1061 unchanged candidates,427 recipe families; original v1/failure retained',flush=True)


if __name__=='__main__':
    main()
