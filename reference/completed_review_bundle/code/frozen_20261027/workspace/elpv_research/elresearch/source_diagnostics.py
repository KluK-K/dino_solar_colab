"""Post-hoc development diagnostics for source/label association, not target fitting."""
import numpy as np
from sklearn.metrics import roc_auc_score


def source_summary(y,score,sources):
    y=np.asarray(y);score=np.asarray(score);sources=np.asarray(sources)
    if y.shape!=score.shape or y.shape!=sources.shape or not np.isfinite(score).all() or not set(np.unique(y))<={0,1}:
        raise ValueError('Aligned finite source diagnostic inputs required')
    by_source={}
    for source in sorted(set(sources.tolist())):
        keep=sources==source
        by_source[source]={'images':int(keep.sum()),'positive':int(y[keep].sum()),
            'auroc':float(roc_auc_score(y[keep],score[keep])) if len(set(y[keep]))==2 else None}
    eligible=[r['auroc'] for r in by_source.values() if r['auroc'] is not None]
    return {'pooled_auroc':float(roc_auc_score(y,score)) if len(set(y))==2 else None,
            'macro_within_source_auroc':float(np.mean(eligible)) if eligible else None,
            'eligible_two_class_sources':len(eligible),'by_source':by_source}


def source_prevalence_scores(training,validation,smoothing=1.):
    if not training or smoothing<=0:raise ValueError('Training rows and positive smoothing required')
    if any(r['partition']!='train' for r in training):raise PermissionError('Source prior can only fit selected training labels')
    global_rate=(sum(r['target'] for r in training)+smoothing)/(len(training)+2*smoothing)
    counts={}
    for row in training:
        group=counts.setdefault(row['acquisition_source'],[0,0]);group[0]+=row['target'];group[1]+=1
    priors={name:(n+smoothing)/(d+2*smoothing) for name,(n,d) in counts.items()}
    return np.array([priors.get(r['acquisition_source'],global_rate) for r in validation]),{
        'training_priors':priors,'unseen_source_fallback':global_rate,'smoothing':smoothing,
        'scope':'Metadata-only diagnostic, not a deployment image classifier; no validation labels fit the priors'}
