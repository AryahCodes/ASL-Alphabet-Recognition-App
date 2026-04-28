"""
test_frame_buffer.py — unit tests for FrameBuffer (defined in server.py).
Imported via conftest fixtures.
"""

import sys
import os
import pytest

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from server import FrameBuffer


def _lm(val=0.5):
    """Return 21 synthetic landmarks all set to val."""
    return [{"x": val + i * 0.01, "y": val + i * 0.005, "z": 0.0} for i in range(21)]


class TestFrameBufferReadiness:
    def test_not_ready_on_init(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        assert not buf.is_ready()

    def test_not_ready_below_min_frames(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(4):
            buf.add_frame(_lm())
        assert not buf.is_ready()

    def test_ready_at_min_frames(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(5):
            buf.add_frame(_lm())
        assert buf.is_ready()

    def test_ready_above_min_frames(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(8):
            buf.add_frame(_lm())
        assert buf.is_ready()


class TestShouldPredict:
    def test_should_predict_false_below_min_frames(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(3):
            buf.add_frame(_lm())
        assert not buf.should_predict()

    def test_should_predict_true_at_first_eligible_frame(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        result = None
        for _ in range(5):
            buf.add_frame(_lm())
        # Frame count is now 5 — should trigger prediction
        result = buf.should_predict()
        assert result is True

    def test_should_predict_false_immediately_after_prediction(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(5):
            buf.add_frame(_lm())
        buf.should_predict()  # consume the prediction
        assert not buf.should_predict()

    def test_should_predict_true_after_5_more_frames(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(5):
            buf.add_frame(_lm())
        buf.should_predict()  # first prediction
        for _ in range(5):
            buf.add_frame(_lm())
        assert buf.should_predict()


class TestGetAverageLandmarks:
    def test_returns_none_when_not_ready(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        assert buf.get_average_landmarks() is None

    def test_average_is_correct(self):
        buf = FrameBuffer(buffer_size=10, min_frames=2)
        # Frame A: all x=0.0, Frame B: all x=1.0 → average should be 0.5
        lm_a = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]
        lm_b = [{"x": 1.0, "y": 1.0, "z": 0.0} for _ in range(21)]
        buf.add_frame(lm_a)
        buf.add_frame(lm_b)
        avg = buf.get_average_landmarks()
        assert avg is not None
        assert len(avg) == 21
        for lm in avg:
            assert abs(lm["x"] - 0.5) < 1e-6
            assert abs(lm["y"] - 0.5) < 1e-6

    def test_returns_21_landmarks(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(5):
            buf.add_frame(_lm())
        avg = buf.get_average_landmarks()
        assert len(avg) == 21


class TestClear:
    def test_clear_resets_readiness(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(7):
            buf.add_frame(_lm())
        assert buf.is_ready()
        buf.clear()
        assert not buf.is_ready()

    def test_clear_resets_predict_cadence(self):
        buf = FrameBuffer(buffer_size=10, min_frames=5)
        for _ in range(5):
            buf.add_frame(_lm())
        buf.should_predict()
        buf.clear()
        for _ in range(5):
            buf.add_frame(_lm())
        # After clear + refill, should be able to predict again
        assert buf.should_predict()
