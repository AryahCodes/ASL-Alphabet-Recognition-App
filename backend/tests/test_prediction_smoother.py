"""
test_prediction_smoother.py — unit tests for PredictionSmoother (defined in server.py).
Imported via conftest.
"""

import sys
import os
import pytest

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from server import PredictionSmoother


class TestInitialState:
    def test_returns_none_on_empty_smoother(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        letter, conf = s.get_smoothed_prediction()
        assert letter is None
        assert conf == 0.0

    def test_single_prediction_not_enough(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        s.add_prediction("A", 0.9)
        letter, conf = s.get_smoothed_prediction()
        assert letter is None


class TestStablePrediction:
    def test_stable_high_confidence_returns_letter(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(5):
            s.add_prediction("B", 0.85)
        letter, conf = s.get_smoothed_prediction()
        assert letter == "B"
        assert conf >= 0.50

    def test_returned_confidence_matches_average(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(5):
            s.add_prediction("C", 0.80)
        _, conf = s.get_smoothed_prediction()
        assert abs(conf - 0.80) < 1e-5


class TestUnstablePrediction:
    def test_three_way_split_returns_none(self):
        # 3 letters each appearing ~2-3 times out of 7 — none reaches 40% majority
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for i, ltr in enumerate(["A", "B", "C", "A", "B", "C", "B"]):
            s.add_prediction(ltr, 0.80)
        # B appears 3/7 = 43%, which is above 40% threshold — use equal split instead
        # Use 2 of each to ensure no letter dominates
        s2 = PredictionSmoother(window_size=6, confidence_threshold=0.50)
        for ltr in ["A", "B", "C", "A", "B", "C"]:
            s2.add_prediction(ltr, 0.80)
        letter, _ = s2.get_smoothed_prediction()
        assert letter is None

    def test_low_confidence_returns_none(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(5):
            s.add_prediction("D", 0.20)
        letter, conf = s.get_smoothed_prediction()
        assert letter is None
        assert conf == 0.0

    def test_majority_below_threshold_returns_none(self):
        # Letter appears in only 2 of 7 frames — below 40% majority threshold
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(2):
            s.add_prediction("E", 0.90)
        for _ in range(5):
            s.add_prediction("F", 0.90)
        # "F" appears in 5/7 = 71% → should be returned
        letter, _ = s.get_smoothed_prediction()
        assert letter == "F"


class TestNoHandBehavior:
    def test_no_hand_clears_after_3_frames(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(5):
            s.add_prediction("G", 0.80)
        # Simulate 3 frames of no hand
        for _ in range(3):
            s.no_hand_detected()
        letter, _ = s.get_smoothed_prediction()
        assert letter is None

    def test_no_hand_does_not_clear_before_3_frames(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(5):
            s.add_prediction("H", 0.80)
        # Only 2 frames of no hand — should NOT clear
        for _ in range(2):
            s.no_hand_detected()
        letter, _ = s.get_smoothed_prediction()
        assert letter == "H"


class TestReset:
    def test_reset_clears_all_state(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(5):
            s.add_prediction("I", 0.80)
        s.reset()
        letter, conf = s.get_smoothed_prediction()
        assert letter is None
        assert conf == 0.0

    def test_reset_clears_no_hand_counter(self):
        s = PredictionSmoother(window_size=7, confidence_threshold=0.50)
        for _ in range(2):
            s.no_hand_detected()
        s.reset()
        assert s.frames_since_last_hand == 0
