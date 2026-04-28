# Benchmarking Guide

The backend exposes a `/metrics` endpoint that accumulates live counters and latency stats while the server is running. No separate profiling tool is needed.

## Reading the metrics endpoint

With the server running (`python backend/server.py`), call:

```bash
curl http://localhost:5001/metrics
```

Example response:

```json
{
  "frames_received": 1240,
  "frames_processed": 248,
  "frames_no_hand": 310,
  "frames_failed": 0,
  "active_clients": 1,
  "inference_latency_ms": {
    "count": 248,
    "mean": 12.4,
    "min": 8.1,
    "max": 31.7,
    "p95": 22.3,
    "median": 11.2
  },
  "uptime_seconds": 124.5
}
```

### What each field means

| Field | Description |
|---|---|
| `frames_received` | Total `process_frame` Socket.IO events received |
| `frames_processed` | Frames where MediaPipe found a hand **and** the buffer ran inference |
| `frames_no_hand` | Frames where MediaPipe found no hand |
| `frames_failed` | Frames that raised an exception (decode error, etc.) |
| `active_clients` | Current number of connected Socket.IO sessions |
| `inference_latency_ms.mean` | Average time for one `classifier.predict()` call |
| `inference_latency_ms.p95` | 95th-percentile latency over the last 200 inferences |
| `inference_latency_ms.median` | 50th-percentile (median) latency over the last 200 inferences |

### Derived metrics

```
frames_per_second     = frames_received / uptime_seconds
inference_fps         = frames_processed / uptime_seconds
skip_rate             = 1 - (frames_processed / frames_received)
no_hand_rate          = frames_no_hand / frames_received
failed_rate           = frames_failed / frames_received
```

## Automated benchmark script

`benchmark_client.py` (repo root) connects to the running backend, sends synthetic frames, waits, then prints a summary table.

```bash
# Start the server first
python backend/server.py &

# Run benchmark (default: 200 frames)
python benchmark_client.py

# Custom frame count and server URL
python benchmark_client.py --frames 500 --url http://localhost:5001
```

### Saving benchmark reports

By default, each benchmark run saves a timestamped JSON report:

    python benchmark_client.py
    # → benchmark_results/benchmark_20260427T120000Z.json

To disable saving:

    python benchmark_client.py --no-save

To annotate the run with environment notes:

    python benchmark_client.py --env "M2 MacBook, Python 3.11, gunicorn"

The saved JSON includes all /metrics counters, `processed_fps`, `failure_rate_pct`,
and full latency stats (mean, median, min, max, p95). JSON files are excluded from
git; only the `benchmark_results/.gitkeep` placeholder is tracked.

## Collecting metrics for a resume claim

1. Start the server and open the frontend in a browser.
2. Hold your hand in front of the webcam for ~60 seconds across several letters.
3. Call `curl http://localhost:5001/metrics` and record the output.
4. Use the derived formulas above to compute your actual measured values.

Only cite numbers you measured yourself in a live session — do not use placeholder values.
