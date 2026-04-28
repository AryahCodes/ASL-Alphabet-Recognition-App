"""
Tests for smoothing_ablation.py (repo root).

Run from backend/ with: pytest tests/test_smoothing_ablation.py -v
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from smoothing_ablation import _smooth, _no_smooth, _compute_metrics, run_ablation


def test_stable_sequence_high_stable_pct():
    # 21 frames of "A" at 0.85 — once the window fills all outputs are "A"
    seq = [("A", 0.85)] * 21
    outputs = _smooth(seq, window_size=7)
    metrics = _compute_metrics(outputs)
    assert metrics["stable_pct"] > 80


def test_alternating_no_majority_window3():
    # window=3 → min_majority = max(3, 3*0.4) = 3 (unanimous).
    # Alternating A/B never achieves 3 identical letters in a 3-frame window.
    seq = [("A", 0.85) if i % 2 == 0 else ("B", 0.85) for i in range(35)]
    outputs = _smooth(seq, window_size=3)
    metrics = _compute_metrics(outputs)
    assert metrics["stable_pct"] == 0.0


def test_flicker_rate_zero_for_stable_output():
    # All non-None outputs are "A" — no adjacent changes, so flicker_rate == 0.0
    seq = [("A", 0.85)] * 21
    outputs = _smooth(seq, window_size=7)
    metrics = _compute_metrics(outputs)
    assert metrics["flicker_rate"] == 0.0


def test_flicker_rate_formula():
    # Known output: A, B, A, None, A
    # Non-None letters: [A, B, A, A]
    # Adjacent pairs: (A,B)=change, (B,A)=change, (A,A)=no change → 2/3 changes
    outputs = [("A", 0.8), ("B", 0.8), ("A", 0.8), (None, 0.0), ("A", 0.8)]
    metrics = _compute_metrics(outputs)
    expected = round(2 / 3, 4)
    assert metrics["flicker_rate"] == expected
