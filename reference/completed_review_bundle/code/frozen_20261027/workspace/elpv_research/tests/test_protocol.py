"""Network-free contract tests; random tensors are fixtures, never research results."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from elresearch import data
from elresearch.common import digest
from elresearch.encoders import (FrozenEncoder, geometry, image_tensor, patch_geometry,
                                 patch_targets, split_tokens, reproject, cache_path)
from elresearch.heads import NormalBank, InspectionHead, budget_rows
from elresearch.metrics import classification, recall_threshold, grouped_bootstrap
from elresearch.probes import hog_stats

def test_register_offsets_and_wrong_count():
    tokens = torch.arange(1 + 4 + 9).reshape(1, -1, 1).float()
    cls, patches = split_tokens(tokens, 3, 4)
    assert cls.item() == 0
    assert patches.flatten().tolist() == list(range(5, 14))
    with pytest.raises(ValueError):
        split_tokens(tokens, 3, 0)

def test_thin_crack_survives_downsampling():
    mask = np.zeros((600, 600), np.uint8); mask[301, 307] = 1
    labels = patch_targets(mask, geometry(600, 600, 224), 16)
    assert (labels == 1).any()

def test_unknown_is_not_a_normal_reference():
    mask = np.zeros((32, 32), np.uint8); mask[:16, :16] = data.IGNORE
    assert patch_targets(mask, geometry(32, 32, 32), 16)[0, 0] == data.IGNORE
    with pytest.raises(ValueError, match='Unexpected UCF'):
        data.canonical_mask(np.array([[0, 128]], np.uint8), 'ucf')

def test_label_priority_preserves_known_crack():
    mask = np.full((32, 32), data.IGNORE, np.uint8); mask[3, 3] = 1
    labels = patch_targets(mask, geometry(32, 32, 32), 16)
    assert labels[0, 0] == 1
    assert labels[1, 1] == data.IGNORE

def test_whole_image_letterbox_and_native_map():
    tensor, g = image_tensor(Image.new('L', (80, 40), 128), 64)
    valid, coords = patch_geometry(g, 16)
    assert tensor.shape == (3, 64, 64)
    assert valid.sum() == 8 and g['top'] == 16
    assert reproject(np.ones((4, 4)), g).shape == (40, 80)
    assert np.allclose(reproject(np.ones((4, 4)), g), 1)
    assert coords.shape == (4, 4, 2)

def test_duplicates_join_capture_groups_before_splitting():
    rows = [{'group_id': 'A', 'pixel_sha256': 'x'}, {'group_id': 'B', 'pixel_sha256': 'x'}, {'group_id': 'C', 'pixel_sha256': 'y'}]
    data._connected_groups(rows)
    assert rows[0]['split_group'] == rows[1]['split_group']
    assert rows[2]['split_group'] != rows[0]['split_group']

def test_group_splits_are_disjoint_and_repeatable():
    rows = [{'id': f'{g}/{c}', 'target': c, 'split_group': str(g)} for g in range(30) for c in [0, 1]]
    one = data.make_splits([r.copy() for r in rows], 11)
    two = data.make_splits([r.copy() for r in rows], 11)
    assert one == two
    groups = [{r['split_group'] for r in one if r['partition'] == p} for p in ['train', 'validation', 'test']]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

def test_sealed_loader_and_course_test_guard(monkeypatch):
    monkeypatch.setattr(data, 'read_json', lambda _: {'records': []})
    with pytest.raises(PermissionError):
        data.records('ucf', 'test')
    with pytest.raises(PermissionError):
        data.records('scdd', 'external')
    with pytest.raises(PermissionError):
        data.records('elpv', 'test', evaluation=True)

def test_reference_bank_refuses_validation_before_loading():
    with pytest.raises(PermissionError):
        NormalBank.build([{'partition': 'validation'}], SimpleNamespace())

def test_reference_excludes_query_acquisition_group():
    item = {'cls': torch.tensor([1., 0.]), 'patch': torch.tensor([[1., 0.]]), 'coords': torch.tensor([[.5, .5]]), 'group': 'A'}
    bank = NormalBank([item], 'fixture')
    feature = {'cls': np.array([1., 0.], np.float32), 'patch': np.array([[[1., 0.]]], np.float32), 'coords': np.array([[[.5, .5]]], np.float32)}
    with pytest.raises(ValueError, match='self-reference'):
        bank.residual(feature, query_group='A')
    out = bank.residual(feature, query_group='B')
    assert out.shape == (1, 1, 7)
    assert torch.allclose(out[..., 2:4], torch.zeros(1, 1, 2))

def test_head_ignores_invalid_local_tokens():
    torch.manual_seed(11)
    head = InspectionHead(2, hidden=8).eval()
    x = torch.randn(1, 4, 7); cls = torch.randn(1, 2); mean = torch.randn(1, 2)
    valid = torch.tensor([[True, True, False, False]])
    first, _ = head(x, cls, mean, valid)
    x[:, 2:] = 10000
    second, _ = head(x, cls, mean, valid)
    assert torch.allclose(first, second)

def test_budget_is_reproducible_and_source_only():
    rows = [{'id': str(i), 'target': i % 2} for i in range(100)]
    selected = budget_rows(rows, .1, 11)
    assert selected == budget_rows(rows, .1, 11) and len(selected) == 10
    assert {r['target'] for r in selected} == {0, 1}
    with pytest.raises(ValueError):
        budget_rows(rows, 0, 11)

def test_high_recall_threshold_handles_ties():
    y = np.array([1, 1, 0, 0]); scores = np.array([.5, .5, .4, .1])
    threshold = recall_threshold(y, scores, 1.)
    metrics = classification(y, scores, threshold)
    assert threshold < .5 and metrics['recall'] == 1.
    assert metrics['review_rate'] == .5

def test_cache_changes_when_weights_or_image_changes():
    row = {'id': 'fixture', 'image_sha256': 'a'}
    one = SimpleNamespace(name='fixture', key=digest({'weight': 'a'}))
    two = SimpleNamespace(name='fixture', key=digest({'weight': 'b'}))
    assert cache_path(one, row) != cache_path(two, row)
    assert cache_path(one, row) != cache_path(one, {**row, 'image_sha256': 'b'})

def test_dinov2_disabled_before_any_model_download():
    with pytest.raises(PermissionError, match='disabled'):
        FrozenEncoder('dinov2_small')

def test_course_feature_dimension():
    assert hog_stats(Image.new('L', (300, 300), 128)).shape == (1806,)

def test_paired_bootstrap_identical_models():
    result = grouped_bootstrap([0, 1, 0, 1], [.1, .9, .2, .8], [.1, .9, .2, .8], ['A', 'A', 'B', 'B'], repetitions=30)
    assert result['paired_auc_difference_ci95'] == [0., 0.]
