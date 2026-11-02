"""Evaluation-only prevalence/negative-composition scenarios; never fit a target threshold."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score


def inspection_scenarios(crack_scores,other_defect_scores,normal_scores,source_threshold,
                         prevalences=(.01,.05,.10),normal_negative_fractions=(0.,.5,1.)):
    groups=[np.asarray(x,dtype=np.float64) for x in [crack_scores,other_defect_scores,normal_scores]]
    if any(x.ndim!=1 or not len(x) or not np.isfinite(x).all() for x in groups):
        raise ValueError('Need finite scores for crack, other-defect negative, and author-normal groups')
    if not np.isfinite(source_threshold):raise ValueError('Fixed source threshold must be finite')
    if any(not 0<p<1 for p in prevalences) or any(not 0<=q<=1 for q in normal_negative_fractions):
        raise ValueError('Scenario proportions must be valid probabilities')
    positive,hard,normal=groups;tpr=float((positive>source_threshold).mean())
    hard_fpr=float((hard>source_threshold).mean());normal_fpr=float((normal>source_threshold).mean())
    scores=np.concatenate(groups);y=np.r_[np.ones(len(positive)),np.zeros(len(hard)+len(normal))]
    results=[]
    for prevalence in prevalences:
        for normal_fraction in normal_negative_fractions:
            fpr=normal_fraction*normal_fpr+(1-normal_fraction)*hard_fpr
            review=prevalence*tpr+(1-prevalence)*fpr
            weights=np.concatenate([np.full(len(positive),prevalence/len(positive)),
                np.full(len(hard),(1-prevalence)*(1-normal_fraction)/len(hard)),
                np.full(len(normal),(1-prevalence)*normal_fraction/len(normal))])
            results.append({'assumed_crack_prevalence':float(prevalence),
                'assumed_normal_fraction_among_negatives':float(normal_fraction),
                'target_recall':tpr,'mixture_false_positive_rate':fpr,
                'scenario_review_rate':review,
                'scenario_positive_predictive_value':prevalence*tpr/review if review else None,
                'scenario_missed_crack_images_per_1000':1000*prevalence*(1-tpr),
                'scenario_unnecessary_reviews_per_1000':1000*(1-prevalence)*fpr,
                'scenario_auroc':float(roc_auc_score(y,scores,sample_weight=weights)),
                'scenario_average_precision':float(average_precision_score(y,scores,sample_weight=weights))})
    return {'source_selected_threshold':float(source_threshold),'threshold_fitted_on_target':False,
        'group_sizes':{'crack_positive':len(positive),'other_defect_negative':len(hard),'author_normal_negative':len(normal)},
        'observed_target_recall':tpr,'other_defect_false_positive_rate':hard_fpr,
        'author_normal_false_positive_rate':normal_fpr,'scenarios':results,
        'limits':['Prevalences and negative mixtures are assumed scenarios, not measured field prevalence.',
                  'Class-conditional detection rates are assumed to remain fixed when mixing the evaluated groups.',
                  'Scores and the source threshold remain unchanged; target labels define evaluation groups and weights only.',
                  'Scenario precision and workload are conditional illustrations, not deployment guarantees.',
                  'No independence assumption, confidence interval, new classifier, or calibration fit is supplied here.']}
