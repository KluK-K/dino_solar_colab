"""Compare saved source-validation predictions; never encode a held-out image."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from elresearch.common import ROOT, read_json, save_json, digest
from elresearch.data import records, load_image, load_mask
from elresearch.metrics import grouped_bootstrap


def main():
    runs = read_json(ROOT / 'artifacts/full_architectures.json')['runs']
    if len(runs) != 4:
        raise ValueError('Complete the fixed four-architecture comparison first')
    extension_path = ROOT / 'artifacts/full_tuning_extension_results.json'
    if extension_path.exists():
        runs.append(read_json(extension_path)['run'])
    local = next(r for r in runs if r['architecture'] == 'dinov3_decoder'
                 and r['adaptation'] == 'frozen' and not r['reference'])
    reference = next(r for r in runs if r['reference'])
    predictions = {}
    for run in runs:
        with np.load(ROOT / 'artifacts/dense_runs' / run['run_id'] / 'validation_predictions.npz') as f:
            predictions[run['run_id']] = {key: f[key].copy() for key in f.files}
    a, b = predictions[local['run_id']], predictions[reference['run_id']]
    for key in ['ids', 'y', 'groups']:
        if not np.array_equal(a[key], b[key]):
            raise ValueError('Paired comparison requires identical images, labels and groups')
    comparison = {'first': local['run_id'], 'second': reference['run_id'],
                  'auc_delta': float(roc_auc_score(a['y'], b['scores']) - roc_auc_score(a['y'], a['scores'])),
                  **grouped_bootstrap(a['y'], a['scores'], b['scores'], a['groups'], repetitions=2000)}
    save_json(ROOT / 'artifacts/full_development_bootstrap.json', {
        'scope': 'Descriptive conditional interval after checkpoint selection on reused source validation; not confirmatory.',
        'dense_local_to_reference': comparison})

    from publication_figures import render_heatmaps
    render_heatmaps()
    print(comparison)


if __name__ == '__main__':
    main()
