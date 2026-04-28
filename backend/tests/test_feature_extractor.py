"""
test_feature_extractor.py — unit tests for FeatureExtractor.

FeatureExtractor has no eventlet/mediapipe/tensorflow dependency so it can
be imported directly without any stubs.
"""

import sys
import os
import numpy as np
import pytest

# Ensure backend/ is on the path
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from feature_extractor import FeatureExtractor


def _make_landmarks(n=21, base=0.1, step=0.03):
    """Build n synthetic landmarks with varying x, y, z values."""
    return [
        {"x": base + i * step, "y": base + i * step * 0.7, "z": 0.01 * i}
        for i in range(n)
    ]


@pytest.fixture
def extractor():
    return FeatureExtractor()


class TestExtractFeatures:
    def test_output_shape_is_78(self, extractor):
        lms = _make_landmarks()
        result = extractor.extract_features(lms)
        assert result is not None
        assert result.shape == (72,)

    def test_output_dtype_is_float32(self, extractor):
        lms = _make_landmarks()
        result = extractor.extract_features(lms)
        assert result.dtype == np.float32

    def test_returns_none_for_none_input(self, extractor):
        assert extractor.extract_features(None) is None

    def test_returns_none_for_wrong_landmark_count(self, extractor):
        assert extractor.extract_features(_make_landmarks(10)) is None
        assert extractor.extract_features(_make_landmarks(22)) is None
        assert extractor.extract_features([]) is None

    def test_no_nan_or_inf_in_output(self, extractor):
        lms = _make_landmarks()
        result = extractor.extract_features(lms)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_constant_x_coords_do_not_crash(self, extractor):
        """All x the same → std=0; the extractor must guard against division by zero."""
        lms = [{"x": 0.5, "y": 0.1 + i * 0.02, "z": 0.0} for i in range(21)]
        result = extractor.extract_features(lms)
        assert result is not None
        assert result.shape == (72,)

    def test_constant_y_coords_do_not_crash(self, extractor):
        lms = [{"x": 0.1 + i * 0.02, "y": 0.5, "z": 0.0} for i in range(21)]
        result = extractor.extract_features(lms)
        assert result is not None
        assert result.shape == (72,)

    def test_two_different_shapes_produce_different_features(self, extractor):
        # Pose A: landmarks increase linearly (extended hand)
        lms_a = [{"x": 0.1 + i * 0.03, "y": 0.5, "z": 0.0} for i in range(21)]
        # Pose B: landmarks follow a curve (bent hand) - different shape after normalization
        lms_b = [{"x": 0.5 + 0.3 * np.sin(i * 0.3), "y": 0.5 + 0.3 * np.cos(i * 0.3), "z": 0.0}
                 for i in range(21)]
        fa = extractor.extract_features(lms_a)
        fb = extractor.extract_features(lms_b)
        assert not np.allclose(fa, fb)


class TestExtractBatch:
    def test_batch_returns_2d_array(self, extractor):
        batch = [_make_landmarks() for _ in range(5)]
        result = extractor.extract_batch(batch)
        assert result is not None
        assert result.shape == (5, 72)

    def test_batch_skips_invalid_entries(self, extractor):
        batch = [_make_landmarks(), _make_landmarks(10), _make_landmarks()]
        result = extractor.extract_batch(batch)
        assert result.shape == (2, 72)

    def test_batch_all_invalid_returns_none(self, extractor):
        result = extractor.extract_batch([_make_landmarks(5), None])
        assert result is None
