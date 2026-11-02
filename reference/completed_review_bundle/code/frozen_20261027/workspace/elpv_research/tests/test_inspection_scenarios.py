import numpy as np
import pytest

from elresearch.inspection_scenarios import inspection_scenarios


def test_negative_mixture_reveals_easy_normal_but_hard_other_defect_false_alarms():
    result=inspection_scenarios([.8,.9],[.7,.2],[.1,.2],.5,prevalences=[.1],normal_negative_fractions=[0,1])
    hard,normal=result['scenarios']
    assert hard['target_recall']==normal['target_recall']==1.
    assert hard['mixture_false_positive_rate']==.5 and normal['mixture_false_positive_rate']==0.
    assert hard['scenario_review_rate']==.55 and normal['scenario_review_rate']==.1
    assert result['source_selected_threshold']==.5 and not result['threshold_fitted_on_target']


def test_auroc_is_prevalence_invariant_but_precision_changes():
    result=inspection_scenarios([.3,.8],[.4,.9],[.1,.5],.5,prevalences=[.01,.1],normal_negative_fractions=[.5])
    low,high=result['scenarios']
    assert np.isclose(low['scenario_auroc'],high['scenario_auroc'])
    assert low['scenario_positive_predictive_value']<high['scenario_positive_predictive_value']
    assert low['scenario_average_precision']<high['scenario_average_precision']


def test_scenarios_cannot_fit_around_missing_classes_or_nonfinite_scores():
    with pytest.raises(ValueError):inspection_scenarios([],[-1.],[-1.],0.)
    with pytest.raises(ValueError):inspection_scenarios([np.nan],[-1.],[-1.],0.)
    with pytest.raises(ValueError):inspection_scenarios([1.],[-1.],[-1.],0.,prevalences=[1.])
