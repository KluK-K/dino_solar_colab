import numpy as np
import pytest
from elresearch.source_diagnostics import source_summary,source_prevalence_scores


def test_source_only_predictions_can_score_high_without_within_source_discrimination():
    y=np.array([0]*9+[1]+[0]+[1]*9)
    sources=np.array(['A']*10+['B']*10)
    scores=np.array([.1]*10+[.9]*10)
    result=source_summary(y,scores,sources)
    assert result['pooled_auroc']==pytest.approx(.9)
    assert result['macro_within_source_auroc']==.5
    training=[{'partition':'train','target':int(label),'acquisition_source':source} for label,source in zip(y,sources)]
    predicted,fit=source_prevalence_scores(training,[{'acquisition_source':'A'},{'acquisition_source':'unseen'}])
    np.testing.assert_allclose(predicted,[2/12,.5])
    with pytest.raises(PermissionError):source_prevalence_scores([{**training[0],'partition':'test'}],[])
