"""Converged supervised probes; image-only source fitting and source-only operating points."""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from threadpoolctl import threadpool_limits

from .common import ROOT,digest,now,read_json,save_json,sha256
from .image_classification import assert_source_partitions
from .metrics import classification,recall_threshold
from .publication_train import budget_rows,calibration,probability_metrics


def estimator(spec,dimensions,max_iter=50000):
    kwargs={'C':spec['C'],'class_weight':None,'tol':1e-6}
    if spec['classifier']=='logistic':
        model=LogisticRegression(**kwargs,solver='liblinear',max_iter=max_iter,random_state=0)
    elif spec['classifier']=='linear_svm':
        model=LinearSVC(**kwargs,dual=False,max_iter=max_iter,random_state=0)
    elif spec['classifier']=='rbf_svm':
        model=SVC(**kwargs,kernel='rbf',gamma=spec['gamma_factor']/dimensions,cache_size=256,max_iter=-1)
    else:raise ValueError(spec['classifier'])
    return Pipeline([('scale',StandardScaler()),('classifier',model)]) if spec['representation']=='hog_stats' else Pipeline([('classifier',model)])


def fit_probe(spec,bank,training,validation,seed=11,budget=1.,recall=.95):
    assert_source_partitions(training,validation)
    selected=budget_rows(training,budget,seed)
    x=bank.get(selected,augmented=True);y=np.repeat([r['target'] for r in selected],8)
    val_x=bank.get(validation);val_y=np.array([r['target'] for r in validation])
    if set(np.unique(y))!={0,1}:raise ValueError('Probe training lacks a class')
    started=time.perf_counter();attempts=[]
    for maximum in [50000,200000]:
        model=estimator(spec,bank.dimensions,maximum)
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
            warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y)
        messages=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        attempts.append({'maximum_iterations':maximum,'convergence_warnings':messages})
        if not messages:break
    if messages:raise RuntimeError('Probe did not converge after registered retry: '+str(attempts))
    with threadpool_limits(limits=2):scores=np.asarray(model.decision_function(val_x),dtype=np.float64)
    threshold=recall_threshold(val_y,scores,recall);temperature=calibration(val_y,scores)
    result={'completed_at':now(),'specification':spec,'seed':seed,'training_budget_fraction':budget,
        'training_ids':[r['id'] for r in selected],'validation_ids':[r['id'] for r in validation],
        'training_images':len(selected),'augmented_training_views':len(x),'validation_images':len(validation),
        'total_labeled_images_including_validation':len(selected)+len(validation),
        'training_capture_groups':len({r['split_group'] for r in selected}),
        'feature_identity':bank.identity,'feature_bank_sha256':bank.feature_sha256,
        'dimensions':bank.dimensions,'solver_attempts':attempts,
        'solver_iterations':model.named_steps['classifier'].n_iter_.tolist() if hasattr(model.named_steps['classifier'].n_iter_,'tolist') else int(model.named_steps['classifier'].n_iter_),
        'fit_and_source_calibration_seconds':time.perf_counter()-started,
        'timing_limit':'CPU fitting/calibration only; excludes frozen-encoder or HOG cache preparation; concurrent local work may be active; not an end-to-end efficiency claim',
        'source_calibrated_margin_threshold':threshold,'source_temperature':temperature,
        'validation_metrics':classification(val_y,scores,threshold),
        'validation_metrics_at_zero_margin':classification(val_y,scores,0.),
        'source_temperature_metrics':probability_metrics(val_y,scores,temperature),
        'supervision':'source image labels only; no pixel loss or nearest-neighbor classifier',
        'preprocessing_fit':'HOG StandardScaler fitted only on selected source training views; DINO normalization is per-image and not fitted',
        'seed_interpretation':'seed selects the nested training-image budget; the optimizer is deterministic with fixed random_state=0. At 100%, seed records are not independent optimizer replications.',
        'calibration_interpretation':'temperature-scaled decision margin; SVM margins are not native posterior probabilities'}
    return model,result,scores


def save_and_verify_probe(model,result,scores,bank,validation,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/'model.tmp.joblib';destination=folder/'model.joblib'
    joblib.dump(model,temporary,compress=3);temporary.replace(destination)
    restored=joblib.load(destination)
    with threadpool_limits(limits=2):again=np.asarray(restored.decision_function(bank.get(validation)),dtype=np.float64)
    np.testing.assert_array_equal(again,scores)
    result={**result,'checkpoint_sha256':sha256(destination),'source_reload_maximum_absolute_error':0.,
            'source_reload_verified':True}
    np.savez_compressed(folder/'validation_predictions.npz',ids=np.array([r['id'] for r in validation]),
        y=np.array([r['target'] for r in validation]),logits=scores,groups=np.array([r['split_group'] for r in validation]))
    save_json(folder/'run.json',result)
    return result


def evaluate_target_development(model,result,bank,target,permitted_ids,folder):
    if any(r['partition']!='target_development' or r['id'] not in permitted_ids for r in target):
        raise PermissionError('Only registered non-test development targets are permitted')
    if {r['id'] for r in target}&set(result['training_ids']+result['validation_ids']):
        raise ValueError('Target image participated in source fitting')
    folder=Path(folder)
    # No scaler/estimator/calibrator fit is performed on these data.
    with threadpool_limits(limits=2):scores=np.asarray(model.decision_function(bank.get(target)),dtype=np.float64)
    y=np.array([r['target'] for r in target])
    output={'completed_at':now(),'threshold_fitted_on_target':False,'checkpoint_sha256':result['checkpoint_sha256'],
        'target_ids':[r['id'] for r in target],'metrics':classification(y,scores,result['source_calibrated_margin_threshold']),
        'calibration_source_temperature':probability_metrics(y,scores,result['source_temperature']),
        'scope':'source-fitted classifier and scaler; leave-one-source-out DEVELOPMENT; no final confirmation'}
    np.savez_compressed(folder/'target_development_predictions.npz',ids=np.array([r['id'] for r in target]),
        y=y,logits=scores,groups=np.array([r['split_group'] for r in target]))
    save_json(folder/'target_development.json',output);return output
