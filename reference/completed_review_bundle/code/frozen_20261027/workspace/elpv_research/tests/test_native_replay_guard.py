"""A display export must not silently substitute different native outcomes."""
from copy import deepcopy

import pytest

from export_dense_held_source_maps import compare_native


def native_record():
    return {'images': 1, 'pooled_pixel_average_precision': .25,
            'per_image': [{'id': 'source/example', 'known_pixels': 1000,
                           'predicted_crack_pixels': 21, 'positive_image_dice': .4,
                           'pixel_average_precision': .3, 'crack_positive': True}]}


def test_AP_roundoff_does_not_allow_changed_native_counts():
    expected = native_record(); actual = deepcopy(expected)
    actual['pooled_pixel_average_precision'] += 1e-6
    compare_native(expected, actual)
    actual['per_image'][0]['predicted_crack_pixels'] += 1
    with pytest.raises(RuntimeError, match='count/identity/flag'):
        compare_native(expected, actual)


def test_overlap_does_not_inherit_AP_tolerance():
    expected = native_record(); actual = deepcopy(expected)
    actual['per_image'][0]['positive_image_dice'] += 1e-6
    with pytest.raises(RuntimeError, match='does not replay'):
        compare_native(expected, actual)


def test_case_identity_or_missing_field_cannot_pass_replay():
    expected = native_record(); actual = deepcopy(expected)
    actual['per_image'][0]['id'] = 'another/example'
    with pytest.raises(RuntimeError, match='count/identity/flag'):
        compare_native(expected, actual)
    del actual['per_image'][0]['known_pixels']
    with pytest.raises(RuntimeError, match='schema differs'):
        compare_native(expected, actual)
