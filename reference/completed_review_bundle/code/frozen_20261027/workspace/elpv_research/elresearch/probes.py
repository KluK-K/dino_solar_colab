"""Only source-training folds tune classifiers; source validation calibrates recall."""
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from skimage.feature import hog
from skimage.transform import resize

from .common import ROOT, config, digest, save_json, sha256, read_json, now
from .data import records, load_image
from .encoders import vectors, extract
from .metrics import classification, recall_threshold

def hog_stats(image):
    image = np.asarray(image.convert('L'), np.float64) / 255
    dy, dx = np.gradient(image); gradient = np.hypot(dx, dy)
    features = [image.mean(), image.std(), *np.quantile(image, [.1, .5, .9]),
                (image < .15).mean(), (image < .30).mean(), gradient.mean(), gradient.std(), np.quantile(gradient, .9)]
    for row in np.array_split(image, 4):
        for block in np.array_split(row, 4, axis=1):
            features.extend([block.mean(), block.std()])
    h = hog(resize(image, (128, 128), anti_aliasing=True), orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
    return np.r_[features, h]

def definitions(d, seed):
    cs = [1e-5, 1e-4, .001, .01, .1, 1, 10]
    return {
        'logistic': (make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', max_iter=5000, random_state=seed)), {'logisticregression__C': cs}),
        'linear_svm': (make_pipeline(StandardScaler(), LinearSVC(dual=False, max_iter=10000, random_state=seed)), {'linearsvc__C': cs}),
        'rbf_svm': (make_pipeline(StandardScaler(), SVC(kernel='rbf')), {'svc__C': [.1, 1, 10], 'svc__gamma': [.1 / d, 1 / d, 10 / d]})}

def fit_probes(dataset, encoder=None, representation='concat', seed=11, budget=1., batch_size=8):
    manifest = read_json(ROOT / 'data' / dataset / 'manifest.json')
    train = records(dataset, 'development' if dataset == 'elpv' else 'train')
    if budget != 1:
        from .heads import budget_rows
        train = budget_rows(train, budget, seed)
    validation = None if dataset == 'elpv' else records(dataset, 'validation')
    if encoder:
        extract(dataset, encoder, 'development' if dataset == 'elpv' else 'train', batch_size)
        if validation:
            extract(dataset, encoder, 'validation', batch_size)
        X = vectors(encoder, train, representation)
        V = vectors(encoder, validation, representation) if validation else None
        identity = encoder.identity; name = encoder.name
    else:
        X = np.array([hog_stats(load_image(r)) for r in train])
        V = np.array([hog_stats(load_image(r)) for r in validation]) if validation else None
        identity = {'name': 'hog_stats', 'dimensions': X.shape[1], 'code_sha256': sha256(Path(__file__))}; name = 'hog_stats'
    y = np.array([r['target'] for r in train]); groups = np.array([r['split_group'] for r in train])
    strata = np.array([r.get('stratum', str(r['target'])) for r in train])
    fold_count = min(4, len(set(groups)))
    if fold_count < 2:
        raise ValueError('Too few acquisition groups for model selection')
    folds = list(StratifiedGroupKFold(fold_count, shuffle=True, random_state=seed).split(X, strata, groups))
    for a, b in folds:
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            raise ValueError('A source CV fold lacks a class; expand the cohort')
    output = []
    for classifier, (model, grid) in definitions(X.shape[1], seed).items():
        run_id = digest({'dataset': dataset, 'manifest': manifest['signature'], 'encoder': identity, 'representation': representation,
                         'classifier': classifier, 'seed': seed, 'budget': budget, 'code': sha256(Path(__file__))})[:20]
        folder = ROOT / 'artifacts/runs' / run_id
        if (folder / 'frozen.json').exists():
            raise RuntimeError('Run is frozen; training cannot replace it')
        started = time.perf_counter()
        search = GridSearchCV(model, grid, scoring='roc_auc', cv=folds, n_jobs=1, refit=True, error_score='raise', return_train_score=True)
        search.fit(X, y)
        if validation:
            scores = search.best_estimator_.decision_function(V)
            cal_y = np.array([r['target'] for r in validation]); ids = [r['id'] for r in validation]
            calibration = 'source_validation'
        else:
            scores = cross_val_predict(search.best_estimator_, X, y, cv=folds, method='decision_function')
            cal_y = y; ids = [r['id'] for r in train]
            calibration = 'development_OOF_after_parameter_selection; selection_optimism'
        threshold = recall_threshold(cal_y, scores, config()['review_recall'])
        result = {'run_id': run_id, 'kind': 'probe', 'dataset': dataset, 'task': 'requires_review' if dataset == 'elpv' else 'crack_presence',
                  'encoder': identity, 'encoder_key': encoder.key if encoder else None, 'representation': representation,
                  'classifier': classifier, 'seed': seed, 'budget': budget, 'source_manifest': manifest['signature'],
                  'training_ids': [r['id'] for r in train], 'calibration_ids': ids, 'calibration': calibration,
                  'best_params': search.best_params_, 'cv_auc': float(search.best_score_),
                  'threshold': threshold, 'validation_metrics': classification(cal_y, scores, threshold),
                  'seconds': time.perf_counter() - started, 'created_at': now(), 'config': config(),
                  'publication_eligible': manifest.get('publication_eligible', False)}
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, folder / 'classifier.joblib')
        result['checkpoint_sha256'] = sha256(folder / 'classifier.joblib')
        save_json(folder / 'run.json', result)
        np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array(ids), y=cal_y, scores=scores)
        import pandas as pd
        pd.DataFrame(search.cv_results_).to_csv(folder / 'search.csv', index=False)
        output.append(result)
        print(f'{run_id}: {name}/{classifier} source CV AUC={search.best_score_:.4f}', flush=True)
    return output
