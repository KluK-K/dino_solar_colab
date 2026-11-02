import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, precision_score, recall_score)

def classification(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    if not len(y) or not np.isfinite(score).all():
        raise ValueError('Empty or nonfinite predictions')
    prediction = score > threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    return {'n': len(y), 'positive': int(y.sum()), 'auc': float(roc_auc_score(y, score)) if two_classes else None,
            'average_precision': float(average_precision_score(y, score)) if y.sum() else None,
            'macro_f1': float(f1_score(y, prediction, average='macro', labels=[0, 1], zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)) if two_classes else None,
            'accuracy': float((y == prediction).mean()), 'recall': float(recall_score(y, prediction, zero_division=0)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'review_rate': float(prediction.mean()), 'threshold': float(threshold),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def recall_threshold(y, score, target=.95):
    y, score = np.asarray(y), np.asarray(score)
    if not (y == 1).any() or not np.isfinite(score).all():
        raise ValueError('Recall calibration needs finite scores and positive validation labels')
    candidates = np.r_[np.nextafter(score.min(), -np.inf), np.unique(score)]
    eligible = [float(t) for t in candidates if ((score[y == 1] > t).mean() + 1e-12) >= target]
    return max(eligible)

def grouped_bootstrap(y, first, second, groups, repetitions=1000, seed=11):
    y, first, second, groups = map(np.asarray, [y, first, second, groups])
    rng = np.random.default_rng(seed)
    unique = np.unique(groups); changes = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
        if len(np.unique(y[idx])) == 2:
            changes.append(roc_auc_score(y[idx], second[idx]) - roc_auc_score(y[idx], first[idx]))
    if not changes:
        raise ValueError('Bootstrap samples have no two-class support')
    return {'paired_auc_difference_ci95': np.quantile(changes, [.025, .975]).tolist(),
            'successful_replicates': len(changes), 'group_count': len(unique), 'unit': 'provided acquisition group'}
