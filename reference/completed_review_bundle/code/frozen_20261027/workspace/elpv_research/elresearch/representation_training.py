"""Deterministic source-fitted probes; finite convergence checks and no target fitting."""
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

from .common import now
from .image_classification import assert_source_partitions
from .metrics import classification, recall_threshold
from .publication_train import budget_rows, calibration, probability_metrics


def probe_estimator(spec, dimensions, maximum):
    common = {'C':spec['C'],'class_weight':None,'tol':1e-4}
    if spec['classifier'] == 'logistic':
        classifier = LogisticRegression(**common,solver='liblinear',max_iter=maximum,random_state=0)
    elif spec['classifier'] == 'linear_svm':
        classifier = LinearSVC(**common,dual='auto',max_iter=maximum,random_state=0)
    elif spec['classifier'] == 'rbf_svm':
        classifier = SVC(**common,kernel='rbf',gamma=spec['gamma_factor']/dimensions,
                         max_iter=-1,cache_size=256)
    else: raise ValueError(spec['classifier'])
    return Pipeline([('classifier',classifier)])


def fit_representation_probe(spec,bank,training,validation,seed=11,budget=1.,recall=.95):
    assert_source_partitions(training,validation)
    selected = budget_rows(training,budget,seed)
    x = bank.get(selected,augmented=True); y = np.repeat([r['target'] for r in selected],8)
    val_x = bank.get(validation); val_y = np.array([r['target'] for r in validation])
    if set(np.unique(y)) != {0,1}: raise ValueError('Source training lacks a class')
    started = time.perf_counter(); attempts = []
    for maximum in [10000,50000]:
        model = probe_estimator(spec,bank.dimensions,maximum)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y)
        messages = [str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        attempts.append({'maximum_iterations':maximum,'convergence_warnings':messages})
        if not messages: break
    if messages: raise RuntimeError('Registered representation probe failed convergence: '+str(attempts))
    with threadpool_limits(limits=2): scores=np.asarray(model.decision_function(val_x),np.float64)
    threshold=recall_threshold(val_y,scores,recall);temperature=calibration(val_y,scores)
    iterations=model.named_steps['classifier'].n_iter_
    result={'completed_at':now(),'specification':spec,'seed':seed,'training_budget_fraction':budget,
        'training_ids':[r['id'] for r in selected],'validation_ids':[r['id'] for r in validation],
        'training_images':len(selected),'validation_images':len(validation),'augmented_training_views':len(x),
        'total_labeled_images_including_validation':len(selected)+len(validation),
        'training_capture_groups':len({r['split_group'] for r in selected}),
        'feature_identity':bank.identity,'feature_bank_sha256':bank.feature_sha256,'dimensions':bank.dimensions,
        'solver_attempts':attempts,'solver_iterations':iterations.tolist() if hasattr(iterations,'tolist') else int(iterations),
        'fit_and_source_calibration_seconds':time.perf_counter()-started,
        'timing_limit':'CPU fitting only; excludes frozen feature preparation and is not end-to-end inference time',
        'source_calibrated_margin_threshold':threshold,'source_temperature':temperature,
        'validation_metrics':classification(val_y,scores,threshold),
        'validation_metrics_at_zero_margin':classification(val_y,scores,0.),
        'source_temperature_metrics':probability_metrics(val_y,scores,temperature),
        'supervision':'Source image labels only. No mask, target-fitted statistics, kNN or encoder updates.',
        'preprocessing_fit':'No fitted scaler. Per-image non-affine LayerNorm after source-independent pooling.',
        'seed_interpretation':'Only nested class-stratified training subsets vary; full-budget deterministic fit saved once, seed11.',
        'calibration_interpretation':'Source-validation temperature on decision margins; SVM margins are not native posterior probabilities'}
    return model,result,scores
