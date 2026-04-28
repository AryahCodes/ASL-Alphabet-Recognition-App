"""
benchmark_client.py — send synthetic frames to the running backend and
print a latency/throughput summary from the /metrics endpoint.

Usage:
    python benchmark_client.py [--frames N] [--url URL] [--save/--no-save] [--env NOTES]

Requires: pip install socketio[client] requests
"""

import argparse
import base64
import time
import urllib.request
import json
import sys
import os
import pathlib
import datetime

try:
    import socketio
except ImportError:
    raise SystemExit("Install the client library first:  pip install 'python-socketio[client]'")


# A minimal 1x1 black JPEG in base64 (valid image, no actual hand data)
_BLANK_FRAME = (
    "data:image/jpeg;base64,"
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
    "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
    "/9oADAMBAAIRAxEAPwCwABmX/9k="
)


def build_report(url, metrics, frames_sent, duration_seconds, env_notes):
    """Build a structured benchmark report dict."""
    lat = metrics["inference_latency_ms"]
    frames_received = metrics["frames_received"]
    frames_processed = metrics["frames_processed"]
    frames_failed = metrics["frames_failed"]

    return {
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "frames_sent": frames_sent,
        "duration_seconds": round(duration_seconds, 2),
        "send_rate_fps": round(frames_sent / duration_seconds, 2),
        "frames_received": frames_received,
        "frames_processed": frames_processed,
        "frames_no_hand": metrics["frames_no_hand"],
        "frames_failed": frames_failed,
        "processed_fps": round(frames_processed / duration_seconds, 2),
        "failure_rate_pct": round(frames_failed / max(frames_received, 1) * 100, 2),
        "inference_latency_ms": lat,
        "env_notes": env_notes,
    }


def save_report(report, out_dir="benchmark_results"):
    """Save report as a JSON file in out_dir. Returns the path string."""
    os.makedirs(out_dir, exist_ok=True)
    filename = f"benchmark_{report['timestamp'].replace(':', '').replace('-', '')}.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write(json.dumps(report, indent=2))
    print(f"Report saved to {path}")
    return path


def run_benchmark(url: str, num_frames: int, save: bool = True, env_notes: str = "") -> None:
    sio = socketio.Client()
    received = {"count": 0}

    @sio.on("hand_landmarks")
    def on_result(data):
        received["count"] += 1

    print(f"Connecting to {url} ...")
    try:
        sio.connect(url, transports=["polling"])
    except (ConnectionRefusedError, Exception) as e:
        print(f"Error: Could not connect to {url}")
        print(f"Start the server first:  python backend/server.py")
        sys.exit(1)
    print(f"Connected. Sending {num_frames} frames ...")

    t_start = time.time()
    for _ in range(num_frames):
        sio.emit("process_frame", {"frame": _BLANK_FRAME})
        time.sleep(0.05)  # ~20 FPS cap to avoid overwhelming the server

    # Wait for any remaining responses
    time.sleep(2)
    t_elapsed = time.time() - t_start

    sio.disconnect()

    # Fetch metrics
    with urllib.request.urlopen(f"{url}/metrics", timeout=5) as resp:
        metrics = json.loads(resp.read())

    lat = metrics["inference_latency_ms"]

    print("\n" + "=" * 50)
    print("  BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Frames sent          : {num_frames}")
    print(f"  Responses received   : {received['count']}")
    print(f"  Elapsed time (s)     : {t_elapsed:.1f}")
    print(f"  Send rate (fps)      : {num_frames / t_elapsed:.1f}")
    print()
    print(f"  frames_received      : {metrics['frames_received']}")
    print(f"  frames_processed     : {metrics['frames_processed']}")
    print(f"  frames_no_hand       : {metrics['frames_no_hand']}")
    print(f"  frames_failed        : {metrics['frames_failed']}")
    print()
    if lat['count'] > 0:
        print(f"  Inference latency (ms)")
        print(f"    mean             : {lat['mean']}")
        print(f"    min              : {lat['min']}")
        print(f"    max              : {lat['max']}")
        print(f"    p95              : {lat['p95']}")
    else:
        print("  No inferences recorded (no hands detected in blank frames — expected).")
        print("  Run with a real webcam session to collect latency data.")
    print("=" * 50)
    print()
    print("Tip: for real latency numbers, start the server, open the frontend,")
    print("hold your hand in frame for 60s, then run:  curl", f"{url}/metrics")

    report = build_report(url, metrics, num_frames, t_elapsed, env_notes)
    if save:
        save_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL backend benchmark")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to send")
    parser.add_argument("--url", default="http://localhost:5001", help="Backend URL")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                        help="Save benchmark report to benchmark_results/")
    parser.add_argument("--env", default="", help="Environment notes (e.g. 'M2 MacBook, Python 3.11')")
    args = parser.parse_args()
    run_benchmark(args.url, args.frames, save=args.save, env_notes=args.env)
