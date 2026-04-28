"""
run_real_benchmark.py

Direct inference benchmark: times FeatureExtractor + model.predict() on real
training landmark data. No server, no SocketIO, no MediaPipe image detection.

Usage:
    cd /Users/aryahb/SignApp/backend && python ../run_real_benchmark.py [--n N]

Or from repo root:
    python run_real_benchmark.py [--n N] [--no-save]
"""

import argparse
import datetime
import json
import os
import sys
import time

# Resolve paths relative to this file
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
TRAINING_DATA_DIR = os.path.join(BACKEND_DIR, "training_data")


def load_all_samples():
    """Load all landmark JSON files from training_data/. Returns list of landmark dicts."""
    samples = []
    for letter_dir in sorted(os.listdir(TRAINING_DATA_DIR)):
        full_dir = os.path.join(TRAINING_DATA_DIR, letter_dir)
        if not os.path.isdir(full_dir):
            continue
        for fname in sorted(os.listdir(full_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(full_dir, fname)) as f:
                data = json.load(f)
            samples.append((data["label"], data["landmarks"]))
    return samples


def run_real_benchmark(n_frames=200, save=True):
    # Must run from backend/ so model paths resolve
    original_dir = os.getcwd()
    os.chdir(BACKEND_DIR)
    sys.path.insert(0, BACKEND_DIR)

    try:
        from professional_letter_classifier import ProfessionalLetterClassifier
        import numpy as np

        print("Loading model...")
        clf = ProfessionalLetterClassifier()
        ok = clf.load_model()
        if not ok or not clf.is_trained:
            print("ERROR: Model failed to load. Check backend/models/professional_model.h5")
            sys.exit(1)

        print("Loading training samples...")
        samples = load_all_samples()
        if not samples:
            print("ERROR: No training landmark files found in backend/training_data/")
            sys.exit(1)
        print(f"  {len(samples)} samples across {len(set(l for l,_ in samples))} letters "
              f"({', '.join(sorted(set(l for l,_ in samples)))})")

        # Warm up: 5 passes to let TF/JIT settle
        print("Warming up model (5 passes)...")
        for i in range(5):
            clf.predict(samples[i % len(samples)][1])

        # Benchmark loop
        print(f"Running {n_frames} inference passes...")
        latencies_ms = []
        failed = 0
        t_bench_start = time.perf_counter()

        for i in range(n_frames):
            _, landmarks = samples[i % len(samples)]
            t0 = time.perf_counter()
            result = clf.predict(landmarks)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            if result.get("success", False) and result.get("letter") is not None:
                latencies_ms.append(elapsed_ms)
            else:
                failed += 1

        t_bench_end = time.perf_counter()
        duration_seconds = t_bench_end - t_bench_start

        if not latencies_ms:
            print("ERROR: No successful inferences recorded.")
            sys.exit(1)

        latencies_arr = np.array(latencies_ms)
        lat = {
            "count": len(latencies_arr),
            "mean":   round(float(np.mean(latencies_arr)), 3),
            "median": round(float(np.median(latencies_arr)), 3),
            "min":    round(float(np.min(latencies_arr)), 3),
            "max":    round(float(np.max(latencies_arr)), 3),
            "p95":    round(float(np.percentile(latencies_arr, 95)), 3),
        }

        processed = len(latencies_ms)
        processed_fps = round(processed / duration_seconds, 2)
        failure_rate_pct = round(failed / n_frames * 100, 2)

        # Print results table
        print("\n" + "=" * 55)
        print("  REAL INFERENCE BENCHMARK RESULTS")
        print("  (FeatureExtractor + professional_model.h5)")
        print("=" * 55)
        print(f"  Frames run            : {n_frames}")
        print(f"  Elapsed time (s)      : {duration_seconds:.2f}")
        print(f"  Processed FPS         : {processed_fps}")
        print(f"  Frames processed      : {processed}")
        print(f"  Frames failed         : {failed}")
        print(f"  Failure rate          : {failure_rate_pct}%")
        print(f"  Frames no-hand        : 0  (using known-good landmark data)")
        print()
        print(f"  Inference latency (ms)")
        print(f"    count              : {lat['count']}")
        print(f"    mean               : {lat['mean']}")
        print(f"    median             : {lat['median']}")
        print(f"    min                : {lat['min']}")
        print(f"    max                : {lat['max']}")
        print(f"    p95                : {lat['p95']}")
        print("=" * 55)

        report = {
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "benchmark_type": "direct_inference",
            "note": (
                "Bypasses SocketIO and MediaPipe image detection. "
                "Times FeatureExtractor.extract_features() + model.predict() "
                "on real training landmark JSONs from backend/training_data/."
            ),
            "url": "local",
            "frames_sent": n_frames,
            "duration_seconds": round(duration_seconds, 3),
            "send_rate_fps": processed_fps,
            "frames_received": n_frames,
            "frames_processed": processed,
            "frames_no_hand": 0,
            "frames_failed": failed,
            "processed_fps": processed_fps,
            "failure_rate_pct": failure_rate_pct,
            "inference_latency_ms": lat,
            "env_notes": f"direct inference, professional_model.h5, Python {sys.version.split()[0]}",
        }

        if save:
            out_dir = os.path.join(REPO_ROOT, "benchmark_results")
            os.makedirs(out_dir, exist_ok=True)
            ts = report["timestamp"].replace(":", "").replace("-", "")
            path = os.path.join(out_dir, f"direct_inference_{ts}.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {path}")

        return report

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct inference benchmark for SignApp")
    parser.add_argument("--n", type=int, default=200, help="Number of inference passes (default: 200)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON report")
    args = parser.parse_args()
    run_real_benchmark(n_frames=args.n, save=not args.no_save)
