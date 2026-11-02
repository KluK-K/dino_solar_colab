"""Append a complete source-only inventory without replacing the 19Oct catalog."""
from pathlib import Path
import collections
import hashlib
import json

import build_confirmation_catalog as original

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'confirmation_preparation_v1'
DESTINATION = OUT / 'complete_source_candidate_catalog_20261027_v1.json'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def main():
    assert not (BASE / 'final_confirmation_v1/frozen_registry.json').exists()
    assert not (BASE / 'final_confirmation_v1/observation_opened.json').exists()
    assert not DESTINATION.exists(), 'Never replace a completed candidate inventory'
    historical = [OUT / 'candidate_catalog.json', OUT / 'unreviewed_selection.json']
    before = {str(p): sha(p) for p in historical}
    old_save = original.save_json
    def redirected(path, value):
        assert Path(path) == OUT / 'candidate_catalog.json', 'Unexpected original catalog mutation'
        old_save(DESTINATION, value)
    original.save_json = redirected
    try:
        catalog = original.main()
    finally:
        original.save_json = old_save
    assert all(sha(Path(p)) == value for p, value in before.items())
    assert catalog['all_source_stages_complete'] is True
    assert not catalog['unresolved_source_selections_or_runs'], catalog['unresolved_source_selections_or_runs']
    candidates = catalog['candidates']; assert len({x['id'] for x in candidates}) == len(candidates)
    grouped = collections.defaultdict(list)
    for item in candidates: grouped[item['declaration']['family']].append(item['seed'])
    summary = {'source': ref(Path(__file__).absolute()), 'original_inventory_builder': ref(ROOT / 'build_confirmation_catalog.py'),
               'complete_inventory': ref(DESTINATION), 'original_catalogs_preserved': [ref(p) for p in historical],
               'candidates': len(candidates), 'families': len(grouped),
               'counts_by_source_dataset': dict(collections.Counter(x['source_dataset'] for x in candidates)),
               'counts_by_kind': dict(collections.Counter(x['declaration']['kind'] for x in candidates)),
               'counts_by_fold': dict(collections.Counter(x['fold'] for x in candidates)),
               'family_seed_coverage': dict(grouped), 'unresolved_source_selections_or_runs': [],
               'all_original_source_stages_complete': True, 'source_review_completed': False,
               'final_model_choices_or_registry_created': False, 'new_model_fit_inference_or_final_pixels_read': False,
               'interpretation': 'Complete candidate inventory for scientific review, not automatic permission to score every development variant, independent fit counts or a winning-model selection. Repeated architecture/checkpoint/source roles require explicit accounting in the NEWconcrete reviewed specification. Original19Octcatalog/unreviewedselection bytes remain unchanged.'}
    path = OUT / 'complete_source_candidate_catalog_20261027_v1_verification.json'
    assert not path.exists(); old_save(path, summary)
    print(json.dumps({'verification': ref(path), 'candidates': len(candidates), 'families': len(grouped),
                      'counts_by_kind': summary['counts_by_kind'], 'final_registry_created': False}))


if __name__ == '__main__': main()
