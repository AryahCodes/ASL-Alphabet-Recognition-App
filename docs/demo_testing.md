# Live Demo Session Recording

Repeated live webcam sessions build honest, citable performance evidence that reflects real-world conditions. The table below should be filled in from actual runs with a working webcam and a loaded model. Synthetic benchmark data (blank frames) will not produce real latency numbers because MediaPipe will report no hand detected and inference will never run.

## How to capture a session's metrics

**Step 1** — Start the backend:

```bash
python backend/server.py
```

**Step 2** — Open the frontend and perform live signing for 60–120 seconds across several letters.

**Step 3** — Run the benchmark client to capture `/metrics` into a timestamped JSON report in `benchmark_results/`:

```bash
python benchmark_client.py --env "your machine"
```

**Step 4** — Open the saved JSON report (e.g. `benchmark_results/benchmark_20260427T120000Z.json`) and copy the relevant values into the row below.

**Step 5** — Note any reconnects, dropped frames, or unusual behavior in the `notes` column.

## Session log

| date | session | duration_min | machine | median_latency_ms | p95_latency_ms | processed_fps | failed_frame_pct | reconnects | flicker_rate | notes |
|------|---------|-------------|---------|------------------|---------------|--------------|-----------------|------------|-------------|-------|
<!-- EXAMPLE — replace with real data -->
| 2026-01-15 | 1 | 2.0 | M2 MacBook, Python 3.11 | 11.2 | 19.8 | 2.1 | 0.0 | 0 | 0.04 | All 24 letters tested |
<!-- EXAMPLE — replace with real data -->
| 2026-01-15 | 2 | 1.5 | M2 MacBook, Python 3.11 | 10.9 | 21.3 | 2.0 | 0.8 | 0 | 0.06 | Letters J and Z harder |

## Field reference

| Field | Source |
|-------|--------|
| `median_latency_ms` | `inference_latency_ms.median` in benchmark JSON report |
| `p95_latency_ms` | `inference_latency_ms.p95` in benchmark JSON report |
| `processed_fps` | `processed_fps` in benchmark JSON report |
| `failed_frame_pct` | `failure_rate_pct` in benchmark JSON report |
| `reconnects` | Manual count of Socket.IO reconnect toasts in the frontend |
| `flicker_rate` | Run `python smoothing_ablation.py` with real model output (advanced) |

## When results become citable

After **3 or more sessions** you can cite median and p95 latency and processed FPS as measured values. After **5 or more sessions** you can cite stability and flicker behavior.
