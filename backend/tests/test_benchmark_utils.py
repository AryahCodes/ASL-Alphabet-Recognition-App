"""
test_benchmark_utils.py — unit tests for build_report() in benchmark_client.py.

benchmark_client.py has a top-level `import socketio` (in a try/except that
calls SystemExit on ImportError).  socketio is not installed in CI, so we stub
it in sys.modules before importing the module.
"""

import re
import sys
from unittest.mock import MagicMock

# Stub socketio before importing benchmark_client (not installed in CI).
sys.modules.setdefault("socketio", MagicMock())

# benchmark_client.py lives two directories above this file (project root).
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from benchmark_client import build_report  # noqa: E402

import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metrics():
    return {
        "frames_received": 200,
        "frames_processed": 40,
        "frames_no_hand": 155,
        "frames_failed": 5,
        "inference_latency_ms": {
            "count": 40,
            "mean": 12.4,
            "min": 8.1,
            "max": 31.7,
            "p95": 22.3,
            "median": 11.2,
        },
        "active_clients": 1,
        "uptime_seconds": 10.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_required_keys_present(self, sample_metrics):
        report = build_report("http://localhost:5001", sample_metrics, 200, 10.5, "test")
        required = {
            "timestamp",
            "url",
            "frames_sent",
            "duration_seconds",
            "send_rate_fps",
            "frames_received",
            "frames_processed",
            "frames_no_hand",
            "frames_failed",
            "processed_fps",
            "failure_rate_pct",
            "inference_latency_ms",
            "env_notes",
        }
        assert required <= report.keys()

    def test_failure_rate_pct_calculation(self, sample_metrics):
        # frames_failed=5, frames_received=200  →  5/200*100 = 2.5
        report = build_report("http://localhost:5001", sample_metrics, 200, 10.0, "test")
        assert report["failure_rate_pct"] == 2.5

    def test_processed_fps_calculation(self, sample_metrics):
        # frames_processed=40, duration_seconds=10.0  →  4.0 fps
        report = build_report("http://localhost:5001", sample_metrics, 200, 10.0, "test")
        assert report["processed_fps"] == 4.0

    def test_timestamp_is_iso_format(self, sample_metrics):
        report = build_report("http://localhost:5001", sample_metrics, 200, 10.0, "test")
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", report["timestamp"])

    def test_env_notes_passthrough(self, sample_metrics):
        report = build_report("http://localhost:5001", sample_metrics, 200, 10.0, "M2 MacBook")
        assert report["env_notes"] == "M2 MacBook"

    def test_zero_inference_case(self, sample_metrics):
        zero_lat_metrics = {
            **sample_metrics,
            "inference_latency_ms": {
                "count": 0,
                "mean": None,
                "min": None,
                "max": None,
                "p95": None,
                "median": None,
            },
        }
        report = build_report("http://localhost:5001", zero_lat_metrics, 200, 10.0, "test")
        assert report["inference_latency_ms"]["count"] == 0
        assert report["inference_latency_ms"]["median"] is None

    def test_median_key_gracefully_absent(self, sample_metrics):
        # Simulates an older server response that does not include "median".
        # build_report passes inference_latency_ms through as-is, so it must
        # not crash when the key is missing.
        old_lat = {"count": 10, "mean": 12.0, "min": 8.0, "max": 20.0, "p95": 18.0}
        old_metrics = {**sample_metrics, "inference_latency_ms": old_lat}
        report = build_report("http://localhost:5001", old_metrics, 200, 10.0, "test")
        assert report["inference_latency_ms"].get("median") is None
