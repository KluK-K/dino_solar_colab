"""Development-only ELPV review-label bridge; the course test is never read."""
from pathlib import Path
import gc
import warnings

import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .common import ROOT, read_json, save_json, sha256, now, digest
from .encoders import FrozenEncoder
from .metrics import classification

COURSE = ROOT.parent/'elpv_midterm'


def inputs():
    return {'research_manifest': ROOT/'data/elpv/manifest.json',
        'course_split': COURSE/'data/split_manifest.json', 'course_folds': COURSE/'artifacts/folds.json',
        'course_feature_index': COURSE/'cache/development_features.json',
        'course_features': COURSE/'cache/development_features.npz'}


def checked_development():
    paths = inputs(); manifest = read_json(paths['research_manifest'])
    rows = manifest['records']; split = read_json(paths['course_split'])
    index = read_json(paths['course_feature_index']); folds = read_json(paths['course_folds'])
    identifiers = [r['id'].removeprefix('elpv/') for r in rows]
    if len(rows) != 1968 or len(set(identifiers)) != len(rows):
        raise ValueError('Original 1968 unique development identities required')
    if identifiers != split['development_ids'] or identifiers != index['ids']:
        raise ValueError('ELPV development manifest/cache ordering differs')
    if set(identifiers) & set(split['test_ids']) or len(split['test_ids']) != 656:
        raise PermissionError('Course test identities must remain separate and sealed')
    if any(r['partition'] != 'development' or r['annotation_type'] != 'expert_review_label'
        or r['target'] != int(r['expert_label'] > 0) for r in rows):
        raise PermissionError('ELPV review labels cannot become crack ground truth')
    groups = np.array([r['split_group'] for r in rows]); coverage = np.zeros(len(rows), int)
    for fold in folds:
        a, b = np.array(fold['train']), np.array(fold['validation'])
        if (set(a) & set(b) or set(a) | set(b) != set(range(len(rows)))
            or set(groups[a]) & set(groups[b])):
            raise ValueError('Course outer folds overlap by image or exact-pixel family')
        coverage[b] += 1
    if len(folds) != 4 or not np.all(coverage == 1):
        raise ValueError('Every development image must be held out in exactly one outer fold')
    return rows, folds


def pipeline(classifier, dimensions):
    if classifier == 'logistic':
        model = LogisticRegression(solver='liblinear', max_iter=50000, tol=1e-6, random_state=20261008)
        grid = {'logisticregression__C': [.01, .1, 1., 10.]}
    elif classifier == 'linear_svm':
        model = LinearSVC(dual='auto', max_iter=50000, tol=1e-6, random_state=20261008)
        grid = {'linearsvc__C': [.01, .1, 1., 10.]}
    elif classifier == 'rbf_svm':
        model = SVC(kernel='rbf')
        grid = {'svc__C': [.01, .1, 1., 10.], 'svc__gamma': [.1/dimensions, 1./dimensions, 10./dimensions]}
    else:
        raise ValueError('Unknown fixed classifier')
    return make_pipeline(StandardScaler(), model), grid


def extract_cls(rows, folder):
    """Retain 1968x384 CLS only; never persist hundreds of MB of patch caches."""
    path = Path(folder)/'dinov3_cls.npz'; proof = Path(folder)/'embedding_identity.json'
    if path.exists() and proof.exists():
        identity = read_json(proof)
        if sha256(path) != identity['features_sha256']:
            raise RuntimeError('Saved ELPV embedding cache changed')
        with np.load(path, allow_pickle=False) as values:
            np.testing.assert_array_equal(values['ids'], [r['id'] for r in rows])
            return values['X'].copy()
    encoder = FrozenEncoder('dinov3_small', 336, device='mps'); features = []
    for start in range(0, len(rows), 8):
        images = []
        for row in rows[start:start+8]:
            if sha256(row['image_path']) != row['image_sha256']:
                raise RuntimeError('ELPV source image changed')
            with Image.open(row['image_path']) as image:
                images.append(image.copy())
        values = np.stack([item['cls'] for item in encoder.encode(images)])
        features.append(values)
    X = np.concatenate(features).astype(np.float32)
    if X.shape != (1968, 384) or not np.isfinite(X).all():
        raise ValueError('Frozen Small CLS dimensions differ')
    np.savez_compressed(path, X=X, ids=np.array([r['id'] for r in rows]))
    save_json(proof, {'at': now(), 'encoder': encoder.identity, 'features_sha256': sha256(path),
        'images': len(rows), 'scope': 'ELPV course DEVELOPMENT only; no encoder adaptation or test access'})
    del encoder; gc.collect()
    import torch
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    return X


def fit_all(rows, folds, dino, folder):
    folder = Path(folder)
    with np.load(inputs()['course_features'], allow_pickle=False) as value:
        hog = np.c_[value['stats'], value['hog']]
    if hog.shape != (1968, 1806):
        raise ValueError('Course HOG/statistics cache differs')
    y = np.array([r['target'] for r in rows]); strata = np.array([r['stratum'] for r in rows])
    groups = np.array([r['split_group'] for r in rows]); completed = []
    for feature, X in [('hog_stats', hog), ('dinov3_cls', dino)]:
        for classifier in ['logistic', 'linear_svm', 'rbf_svm']:
            for outer, fold in enumerate(folds):
                destination = folder/'fits'/f'{feature}_{classifier}_outer{outer}'; destination.mkdir(parents=True, exist_ok=True)
                a, b = np.array(fold['train']), np.array(fold['validation'])
                if (destination/'run.json').exists():
                    run = read_json(destination/'run.json')
                    if sha256(destination/'model.joblib') != run['checkpoint_sha256']:
                        raise RuntimeError('Supplementary fitted classifier changed')
                else:
                    inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=20261008+outer).split(X[a], strata[a], groups[a]))
                    model, grid = pipeline(classifier, X.shape[1])
                    search = GridSearchCV(model, grid, scoring='roc_auc', cv=inner, n_jobs=1, error_score='raise', refit=True)
                    # A warning is not silently accepted as a converged control.
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning); search.fit(X[a], y[a])
                    scores = search.best_estimator_.decision_function(X[b])
                    joblib.dump(search.best_estimator_, destination/'model.joblib')
                    model = joblib.load(destination/'model.joblib')
                    np.testing.assert_array_equal(model.decision_function(X[b]), scores)
                    run = {'at': now(), 'feature': feature, 'classifier': classifier, 'outer_fold': outer,
                        'training_ids': [rows[i]['id'] for i in a], 'held_out_development_ids': [rows[i]['id'] for i in b],
                        'inner_folds': [{'train': t.tolist(), 'validation': v.tolist()} for t, v in inner],
                        'best_params': search.best_params_, 'inner_selected_AUROC': float(search.best_score_),
                        'outer_metrics_at_default_zero': classification(y[b], scores, 0.),
                        'checkpoint_sha256': sha256(destination/'model.joblib'), 'fresh_reload_predictions_exact': True,
                        'scope': 'Nested development-only CV. Hyperparameters/scaler use outer training images only; default decision threshold zero, not target-calibrated recall.'}
                    np.savez_compressed(destination/'predictions.npz', ids=np.array([rows[i]['id'] for i in b]), y=y[b], scores=scores)
                    run['predictions_sha256'] = sha256(destination/'predictions.npz'); save_json(destination/'run.json', run)
                if sha256(destination/'predictions.npz') != run['predictions_sha256']:
                    raise RuntimeError('Supplementary outer predictions changed')
                completed.append(run)
                save_json(folder/'status.json', {'at': now(), 'state': 'running_nested_CV', 'completed_runs': len(completed), 'planned_runs': 24})
    save_json(folder/'results.json', {'at': now(), 'fits': completed,
        'primary': 'Per-outer-fold ROC-AUC, paired on the original four held-out development folds; report all fold scores.',
        'limits': ['Review labels are not crack masks; course656 test stays sealed.',
            'Provided exact-pixel groups are not physical modules; no physical-module independence claim.',
            'The older course scores selected hyperparameters on the outer folds. Compare these new matched nested controls, not claim identical validation protocols.',
            'Repeated training on already studied ELPV development data is supplementary exploration, not independent confirmation or paper-score replication.']})


def verify_complete(folder):
    folder = Path(folder); status = read_json(folder/'status.json'); protocol = read_json(folder/'protocol.json')
    if status['state'] != 'complete' or status['completed_runs'] != 24 or status['planned_runs'] != 24:
        raise RuntimeError('Complete all24 supplementary outer-fold fits before the final source review')
    if digest({k:v for k,v in protocol.items() if k!='study_id'})[:20] != protocol['study_id']:
        raise RuntimeError('Supplementary protocol changed')
    for value in protocol['source_bindings']:
        if sha256(value['path']) != value['sha256']: raise RuntimeError('Supplementary source binding changed')
    rows, folds = checked_development(); results = read_json(folder/'results.json')
    actual = {(r['feature'], r['classifier'], r['outer_fold']) for r in results['fits']}
    expected = {(f, c, n) for f in ['hog_stats', 'dinov3_cls'] for c in ['logistic', 'linear_svm', 'rbf_svm'] for n in range(4)}
    if len(results['fits']) != 24 or actual != expected: raise RuntimeError('Supplementary model/fold coverage differs')
    for run in results['fits']:
        fold = folds[run['outer_fold']]; destination = folder/'fits'/f"{run['feature']}_{run['classifier']}_outer{run['outer_fold']}"
        if run != read_json(destination/'run.json') or not run['fresh_reload_predictions_exact']:
            raise RuntimeError('Supplementary fit identity/reload evidence changed')
        if ([rows[i]['id'] for i in fold['train']] != run['training_ids']
            or [rows[i]['id'] for i in fold['validation']] != run['held_out_development_ids']):
            raise RuntimeError('Supplementary outer-fold annotation identities changed')
        for filename, key in [('model.joblib', 'checkpoint_sha256'), ('predictions.npz', 'predictions_sha256')]:
            if sha256(destination/filename) != run[key]: raise RuntimeError('Supplementary classifier/predictions changed')
        with np.load(destination/'predictions.npz', allow_pickle=False) as value:
            np.testing.assert_array_equal(value['ids'], run['held_out_development_ids'])
            np.testing.assert_array_equal(value['y'], [rows[i]['target'] for i in fold['validation']])
            if classification(value['y'], value['scores'], 0.) != run['outer_metrics_at_default_zero']:
                raise RuntimeError('Supplementary saved metrics differ from predictions')
    return {'verified_outer_fits': 24, 'course_test_images_opened': False, 'protocol_id': protocol['study_id'],
        'results_sha256': sha256(folder/'results.json')}
