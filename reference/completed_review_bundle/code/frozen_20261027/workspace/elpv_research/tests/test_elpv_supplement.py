"""Supplement cannot open course test pixels or mix the crack/review tasks."""
import numpy as np
import pytest
from PIL import Image
from elresearch import elpv_supplement as study
from elresearch.common import save_json, sha256, read_json


@pytest.fixture
def course(tmp_path, monkeypatch):
    ids = [f'images/fixture{i}.png' for i in range(1968)]
    rows = [{'id': 'elpv/'+identifier, 'partition': 'development', 'annotation_type': 'expert_review_label',
        'expert_label': float(i % 2), 'target': i % 2, 'split_group': f'unique{i}'} for i, identifier in enumerate(ids)]
    folds = []
    for fold in range(4):
        val = list(range(fold, 1968, 4)); train = sorted(set(range(1968))-set(val)); folds.append({'train': train, 'validation': val})
    paths = {name: tmp_path/(name+'.json') for name in ['research_manifest', 'course_split', 'course_folds', 'course_feature_index', 'course_features']}
    save_json(paths['research_manifest'], {'records': rows})
    save_json(paths['course_split'], {'development_ids': ids, 'test_ids': [f'sealed{i}' for i in range(656)]})
    save_json(paths['course_folds'], folds); save_json(paths['course_feature_index'], {'ids': ids})
    monkeypatch.setattr(study, 'inputs', lambda: paths)
    monkeypatch.setattr(study.Image, 'open', lambda *a: pytest.fail('Metadata audit cannot open any test/development image'))
    return paths


def test_original_outer_folds_have_complete_disjoint_development_coverage(course):
    rows, folds = study.checked_development()
    assert len(rows) == 1968 and len(folds) == 4
    split = read_json(course['course_split']); split['test_ids'][0] = split['development_ids'][0]
    save_json(course['course_split'], split)
    with pytest.raises(PermissionError, match='Course test'): study.checked_development()


def test_crack_labels_or_duplicate_family_crossing_outer_folds_are_rejected(course):
    manifest = read_json(course['research_manifest']); manifest['records'][0]['annotation_type'] = 'crack_mask'
    save_json(course['research_manifest'], manifest)
    with pytest.raises(PermissionError, match='crack ground truth'): study.checked_development()
    manifest['records'][0]['annotation_type'] = 'expert_review_label'
    manifest['records'][0]['split_group'] = manifest['records'][1]['split_group']
    save_json(course['research_manifest'], manifest)
    with pytest.raises(ValueError, match='exact-pixel family'): study.checked_development()


def test_cls_extraction_obeys_list_of_per_image_encoder_outputs_and_exact_cache(tmp_path, monkeypatch):
    image = tmp_path/'observed_development.png'; Image.new('L', (4, 4), 100).save(image)
    rows = [{'id': f'development-fixture{i}', 'image_path': str(image), 'image_sha256': sha256(image)} for i in range(1968)]
    class Encoder:
        identity = {'fixture': 'CPU-only fake encoder, not real model weights'}
        def __init__(self, *a, **k): pass
        def encode(self, images):
            return [{'cls': np.full(384, 3., np.float32), 'patch': np.zeros((1, 1, 384))} for image in images]
    monkeypatch.setattr(study, 'FrozenEncoder', Encoder)
    X = study.extract_cls(rows, tmp_path)
    assert X.shape == (1968, 384) and np.all(X == 3.)
    monkeypatch.setattr(study, 'FrozenEncoder', lambda *a, **k: pytest.fail('Exact cached continuation need not load an encoder'))
    np.testing.assert_array_equal(study.extract_cls(rows, tmp_path), X)
    (tmp_path/'dinov3_cls.npz').write_bytes(b'changed fixture cache')
    with pytest.raises(RuntimeError, match='cache changed'): study.extract_cls(rows, tmp_path)
