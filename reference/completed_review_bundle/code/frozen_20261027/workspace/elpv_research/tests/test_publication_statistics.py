import numpy as np
import pytest

from elresearch.publication_statistics import paired_seed_group_bootstrap


def prediction(seed=11):
    return {'ids':np.arange(12),'y':np.tile([0,1],6),'groups':np.repeat(np.arange(6),2),
            'logits':np.tile([-.4,.4],6)+np.repeat(np.linspace(-.5,.5,6),2),
            'threshold':-.1,'seed':seed}


def test_identical_models_have_exactly_zero_paired_differences():
    rows=[prediction(seed) for seed in [11,22,33]]
    result=paired_seed_group_bootstrap(rows,rows,repetitions=50)
    assert result['acquisition_families']==6 and result['images']==12
    for metric in result['metrics'].values():
        assert metric['mean_seed_difference']==0.
        assert metric['acquisition_family_percentile_ci95']==[0.,0.]


def test_uncertainty_rejects_image_or_seed_misalignment():
    row=prediction();other={**row,'ids':row['ids'][::-1]}
    with pytest.raises(ValueError,match='aligned'):paired_seed_group_bootstrap([row],[other],50)
    with pytest.raises(ValueError,match='seed'):paired_seed_group_bootstrap([row],[prediction(22)],50)


def test_missing_or_single_group_cannot_produce_a_fake_narrow_interval():
    row=prediction();missing={**row,'groups':np.array([None]*12)}
    with pytest.raises(ValueError,match='missing'):paired_seed_group_bootstrap([missing],[missing],50)
    one={**row,'groups':np.zeros(12,dtype=int)}
    with pytest.raises(ValueError,match='two provided'):paired_seed_group_bootstrap([one],[one],50)
