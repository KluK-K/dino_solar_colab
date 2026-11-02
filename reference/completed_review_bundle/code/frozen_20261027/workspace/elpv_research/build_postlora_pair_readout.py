"""Reconstruct descriptive head contrasts from complete saved source readouts.

No model loads, new predictions, bootstrap, thresholds or scientific fits.
The first three-cell snapshot was created from the same explicit rule; this
helper verifies and reuses it rather than replacing its original bytes.
"""
from elresearch.common import ROOT, read_json, save_json, sha256, digest, now

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'mechanistic_complete_cells_v1'
RULE = ('Every complete cell, every3fitted seeds, both fresh-minus-parent and '
        'reference-minus-fresh pairs;6original saved endpoints. Retrospective '
        'descriptive, no CI/pvalue/selection.')
KEYS = ['AUROC', 'AP', 'Dice', 'structure_FPR', 'image_recall', 'image_review_fraction']


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('Development-only readout cannot change after confirmation opens')
    index = read_json(OUT / 'index.json')
    if index['builder_sha256'] != sha256(ROOT / 'build_mechanistic_progress_readout.py'):
        raise RuntimeError('Complete-cell reporter changed')
    pairs = []; lineage = []
    for entry in index['complete_cells']:
        path = ROOT / entry['path']; cell = read_json(path)
        if sha256(path) != entry['sha256']:
            raise RuntimeError('Preserve complete-cell snapshot')
        for seed in [11, 22, 33]:
            families = {r['family']: r for r in cell['all_original_fit_records'] if r['seed'] == seed}
            for family in ['dinov3_postlora_no_reference', 'dinov3_postlora_reference']:
                run_path = ROOT / families[family]['source_run']; run = read_json(run_path)
                parent_path = ROOT / families['dinov3_dense_lora']['source_run']
                if (run['model']['adaptation_parent_run'] != str(parent_path.relative_to(ROOT)) or
                        sha256(parent_path) != run['model']['adaptation_parent_run_sha256'] or
                        run['seed'] != seed or read_json(parent_path)['seed'] != seed):
                    raise RuntimeError('Fresh head is not paired with its actual encoder parent')
                lineage.append({'path': str(run_path.relative_to(ROOT)), 'sha256': sha256(run_path),
                                'parent': str(parent_path.relative_to(ROOT)), 'parent_sha256': sha256(parent_path)})
            for second, first in [('dinov3_postlora_no_reference', 'dinov3_dense_lora'),
                                  ('dinov3_postlora_reference', 'dinov3_postlora_no_reference')]:
                pairs.append({'fold': entry['fold'], 'budget': entry['budget'], 'seed': seed,
                              'second_minus_first': [second, first],
                              'difference': {k: families[second][k] - families[first][k] for k in KEYS},
                              'source_cell': entry})
    identity = {'source_cells': index['complete_cells'], 'comparison_rule': RULE}
    folder = OUT / 'paired_head_readouts' / digest(identity)[:20]
    folder.mkdir(parents=True, exist_ok=True); result_path = folder / 'results.json'
    if result_path.exists():
        original = read_json(result_path)
        if original['identity'] != identity or original['pairs'] != pairs:
            raise RuntimeError('Preserve prior descriptive outcomes')
    else:
        save_json(result_path, {'at': now(), 'identity': identity, 'pairs': pairs,
                  'new_inference': False, 'independent_confirmation': False,
                  'limit': 'Fresh-vs-parent adds training and refits heads; it does not isolate layer depth or equal-compute benefit. Normal-reference matching-specific causality remains unestablished.'})
    save_json(OUT / 'paired_head_readouts/latest.json',
              {'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path)})
    proof_path = folder / ('reconstruction_' + sha256(__file__)[:12] + '.json')
    proof = {'builder_sha256': sha256(__file__), 'result_sha256': sha256(result_path),
             'source_cells': index['complete_cells'], 'exact_parent_lineage_checks': lineage,
             'all_original_pairs_reconstructed': len(pairs), 'new_inference': False,
             'scope': 'Saved source-only descriptive differences, no p-values or independent confirmation.'}
    if proof_path.exists():
        if read_json(proof_path) != proof:
            raise RuntimeError('Preserve reconstruction proof')
    else:
        save_json(proof_path, proof)
    print('Reused/reconstructed complete source head contrasts:', len(pairs),
          'paired records /', len(index['complete_cells']), 'cells; no new inference')


if __name__ == '__main__':
    main()
