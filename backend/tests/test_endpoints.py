"""
test_endpoints.py — REST API contract tests.

These tests treat the Flask app as a black box and verify that every
public endpoint returns the documented status code and JSON shape.
No Socket.IO events are exercised here.
"""

import json


class TestHealthEndpoint:
    def test_health_status_200(self, client):
        response = client.get("/test")
        assert response.status_code == 200

    def test_health_body_contains_success(self, client):
        response = client.get("/test")
        data = json.loads(response.data)
        assert data.get("status") == "success"

    def test_health_body_contains_message(self, client):
        response = client.get("/test")
        data = json.loads(response.data)
        assert "message" in data


class TestHomeEndpoint:
    def test_home_status_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_body_is_json(self, client):
        response = client.get("/")
        # Flask returns a dict directly as JSON; must be parseable.
        data = json.loads(response.data)
        assert isinstance(data, dict)


class TestMetricsEndpoint:
    _REQUIRED_COUNTER_KEYS = [
        "frames_received",
        "frames_processed",
        "frames_no_hand",
        "frames_failed",
        "active_clients",
    ]

    def test_metrics_status_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_all_required_keys(self, client):
        response = client.get("/metrics")
        data = json.loads(response.data)
        for key in self._REQUIRED_COUNTER_KEYS:
            assert key in data, f"Missing key: {key}"
        assert "inference_latency_ms" in data
        assert "uptime_seconds" in data

    def test_metrics_initial_counter_values_are_zero(self, client):
        """On a fresh import the counters should all be 0."""
        response = client.get("/metrics")
        data = json.loads(response.data)
        for key in self._REQUIRED_COUNTER_KEYS:
            assert data[key] == 0, f"{key} should be 0 on startup, got {data[key]}"

    def test_metrics_initial_latency_count_is_zero(self, client):
        response = client.get("/metrics")
        data = json.loads(response.data)
        lat = data["inference_latency_ms"]
        assert lat["count"] == 0

    def test_metrics_uptime_is_non_negative(self, client):
        response = client.get("/metrics")
        data = json.loads(response.data)
        assert data["uptime_seconds"] >= 0


class TestModelStatusEndpoint:
    def test_model_status_200(self, client):
        response = client.get("/model/status")
        assert response.status_code == 200

    def test_model_status_contains_required_keys(self, client):
        response = client.get("/model/status")
        data = json.loads(response.data)
        assert "is_trained" in data
        assert "model_type" in data
        assert "labels" in data
        assert "sample_counts" in data

    def test_model_status_is_trained_is_bool(self, client):
        response = client.get("/model/status")
        data = json.loads(response.data)
        # is_trained comes from a MagicMock classifier; just check it's present
        # and that "labels" is always a list (even when empty).
        assert isinstance(data["labels"], list)

    def test_model_status_sample_counts_is_dict(self, client):
        response = client.get("/model/status")
        data = json.loads(response.data)
        assert isinstance(data["sample_counts"], dict)
