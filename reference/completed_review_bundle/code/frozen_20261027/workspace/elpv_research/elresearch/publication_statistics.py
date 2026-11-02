"""Paired acquisition-family uncertainty, conditional on registered trained seeds."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score


def _metrics(y,logits,threshold,weight):
    pred=logits>threshold;positive=float(weight[y==1].sum());negative=float(weight[y==0].sum())
    return np.array([roc_auc_score(y,logits,sample_weight=weight),
        average_precision_score(y,logits,sample_weight=weight),
        float((weight*pred*y).sum()/positive),float((weight*pred).sum()/weight.sum()),
        float((weight*pred*(1-y)).sum()/negative)])


def paired_seed_group_bootstrap(first,second,repetitions=1000,seed=20261017):
    """Difference second-minus-first, averaged over paired seeds (not an ensemble)."""
    required={'ids','y','groups','logits','threshold','seed'}
    if len(first)!=len(second) or not first:raise ValueError('Missing paired seed support')
    first=sorted(first,key=lambda r:r['seed']);second=sorted(second,key=lambda r:r['seed'])
    for a,b in zip(first,second):
        if not required<=set(a) or not required<=set(b):raise ValueError('Incomplete prediction contract')
        if a['seed']!=b['seed']:raise ValueError('Unpaired training seed')
        for key in ['ids','y','groups']:
            if not np.array_equal(a[key],b[key]) or not np.array_equal(a[key],first[0][key]):
                raise ValueError('Predictions are not aligned by image, label and acquisition group')
        if not np.isfinite(a['logits']).all() or not np.isfinite(b['logits']).all():raise ValueError('Nonfinite logits')
    if len({r['seed'] for r in first})!=len(first):raise ValueError('Duplicate training seed')
    y=np.asarray(first[0]['y']);groups=np.asarray(first[0]['groups'])
    if set(np.unique(y))!={0,1}:raise ValueError('Two classes are required')
    if groups.ndim!=1 or len(groups)!=len(y):raise ValueError('Group vector shape differs from labels')
    for group in groups:
        if group is None or (isinstance(group,(float,np.floating)) and not np.isfinite(group)):
            raise ValueError('Acquisition grouping is missing; do not substitute arbitrary filename blocks')
        if isinstance(group,(str,np.str_)) and (not group.strip() or group.lower() in {'none','unknown','nan'}):
            raise ValueError('Acquisition grouping is missing; do not substitute arbitrary filename blocks')
    unique,inverse=np.unique(groups,return_inverse=True);rng=np.random.default_rng(seed)
    if len(unique)<2:raise ValueError('At least two provided acquisition groups are required for cluster uncertainty')
    names=['auroc','average_precision','recall_at_source_threshold','review_rate_at_source_threshold','false_positive_rate_at_source_threshold']
    def differences(weight):
        return np.array([_metrics(y,np.asarray(b['logits']),b['threshold'],weight)-
                         _metrics(y,np.asarray(a['logits']),a['threshold'],weight)
                         for a,b in zip(first,second)])
    observed=differences(np.ones(len(y)));replicates=[]
    for _ in range(repetitions):
        count=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
        weight=count[inverse]
        if not weight[y==0].sum() or not weight[y==1].sum():continue
        replicates.append(differences(weight).mean(0))
    if len(replicates)<max(20,repetitions//2):raise ValueError('Insufficient two-class cluster replicates')
    intervals=np.quantile(np.asarray(replicates),[.025,.975],axis=0)
    return {'difference':'second minus first; mean of paired per-seed metric differences',
        'seeds':[r['seed'] for r in first],'images':len(y),'acquisition_families':len(unique),
        'requested_repetitions':repetitions,'valid_repetitions':len(replicates),'bootstrap_seed':seed,
        'metrics':{name:{'mean_seed_difference':float(observed[:,i].mean()),
            'paired_seed_differences':observed[:,i].tolist(),
            'seed_difference_sample_sd':float(observed[:,i].std(ddof=1)) if len(first)>1 else None,
            'acquisition_family_percentile_ci95':intervals[:,i].tolist()} for i,name in enumerate(names)},
        'interpretation':['Intervals resample acquisition families, conditional on the fitted models and their registered seeds.',
                          'Training-seed variability is reported separately; the interval is not a population over all possible training runs.',
                          'No target threshold fitting, model ensembling, patch pseudo-replication or adjusted hypothesis test is performed.',
                          'Development comparisons are exploratory; many comparisons must not be described as confirmatory significance.']}
