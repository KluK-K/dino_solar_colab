import importlib.util

import joblib
import numpy as np
import pytest
from skimage.feature import hog
from skimage.transform import resize

from elresearch.common import ROOT
from elresearch.classical_features import FeatureBank,image_statistics,hog_statistics
from elresearch.classical_training import fit_probe,save_and_verify_probe,evaluate_target_development


def rows():
    return [{'id':str(i),'image_sha256':str(i),'pixel_sha256':str(i),'split_group':str(i),
             'partition':'train' if i<16 else 'validation','target':i%2} for i in range(24)]


def test_hog_stats_preserve_course_descriptor_definition():
    module_spec=importlib.util.spec_from_file_location('course_reference',ROOT.parent/'elpv_midterm/data.py')
    reference=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(reference)
    gray=np.random.default_rng(11).integers(0,256,(300,300),dtype=np.uint8)
    np.testing.assert_array_equal(image_statistics(gray),reference.stats_features(gray))
    expected=np.r_[reference.stats_features(gray),hog(resize(gray,(128,128),anti_aliasing=True),
        orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys',transform_sqrt=False)]
    np.testing.assert_allclose(hog_statistics(gray),expected,rtol=1e-6,atol=1e-8)
    assert expected.shape==(1806,)


@pytest.mark.parametrize('classifier',['logistic','linear_svm','rbf_svm'])
def test_source_only_scaler_and_saved_models_reproduce_decisions(tmp_path,classifier):
    data=rows();rng=np.random.default_rng(11);values=rng.normal(size=(24,8,6)).astype(np.float32)
    values[:16]+=np.array([r['target'] for r in data[:16]])[:,None,None]
    values[16:]+=1000  # A validation shift must never enter the fitted source scaler.
    bank=FeatureBank(data,values,{'fixture':'no image-model performance claim'})
    spec={'representation':'hog_stats','classifier':classifier,'C':1.,'gamma_factor':1.}
    model,result,scores=fit_probe(spec,bank,data[:16],data[16:])
    np.testing.assert_allclose(model.named_steps['scale'].mean_,values[:16].reshape(-1,6).mean(0,dtype=np.float64))
    saved=save_and_verify_probe(model,result,scores,bank,data[16:],tmp_path/classifier)
    assert saved['source_reload_verified'] and saved['source_reload_maximum_absolute_error']==0.
    bad=[{**data[-1],'partition':'test'}]
    with pytest.raises(PermissionError):evaluate_target_development(model,saved,bank,bad,{bad[0]['id']},tmp_path/classifier)


def test_feature_reuse_checks_image_identity_and_missing_source_classes():
    data=rows();bank=FeatureBank(data,np.zeros((24,8,6)),{'fixture':'identity check'})
    with pytest.raises(RuntimeError,match='identity'):bank.get([{**data[0],'image_sha256':'changed'}])
