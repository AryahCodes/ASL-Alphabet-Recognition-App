"""
smoothing_ablation.py

Evaluates the effect of different temporal smoothing window sizes on prediction
stability for a sign language recognition app.

Usage:
    python smoothing_ablation.py [--no-save] [--fps 20.0] [--out-dir eval_results]
"""

import argparse
import json
import os
from collections import Counter, deque
from datetime import datetime, timezone
from statistics import mean


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

SEQUENCES = {
    "stable_A": [("A", 0.85)] * 50,
    "noisy_AB": [("A", 0.85) if i % 2 == 0 else ("B", 0.85) for i in range(50)],
}

CONFIGS = [
    {"name": "none (raw)",          "type": "no_smooth"},
    {"name": "window=3",            "type": "smooth", "window": 3},
    {"name": "window=5",            "type": "smooth", "window": 5},
    {"name": "window=7 (current)",  "type": "smooth", "window": 7},
    {"name": "window=10",           "type": "smooth", "window": 10},
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _smooth(sequence, window_size, confidence_threshold=0.50):
    """
    sequence: list of (letter_str, confidence_float)
    Returns: list of (letter_str | None, float)
    Reimplements the app's PredictionSmoother majority-vote logic.
    """
    buf = deque(maxlen=window_size)
    min_majority = max(3, window_size * 0.4)
    results = []
    for letter, conf in sequence:
        buf.append((letter, conf))
        if len(buf) < 2:
            results.append((None, 0.0))
            continue
        counts = Counter(l for l, _ in buf)
        best, best_count = counts.most_common(1)[0]
        if best_count >= min_majority:
            avg_conf = sum(c for l, c in buf if l == best) / best_count
            if avg_conf >= confidence_threshold:
                results.append((best, round(avg_conf, 3)))
                continue
        results.append((None, 0.0))
    return results


def _no_smooth(sequence, confidence_threshold=0.50):
    """Pass-through: return raw prediction if above threshold."""
    return [(l, round(c, 3)) if c >= confidence_threshold else (None, 0.0)
            for l, c in sequence]


def _compute_metrics(outputs, fps=20.0):
    """
    outputs: list of (letter|None, confidence)
    Returns dict with: stable_pct, flicker_rate, changes_per_second, avg_confidence
    """
    total = len(outputs)
    non_none = [(l, c) for l, c in outputs if l is not None]
    stable_pct = round(100.0 * len(non_none) / total, 1) if total > 0 else 0.0

    # flicker: fraction of adjacent non-None pairs that differ
    letters_only = [l for l, _ in non_none]
    if len(letters_only) <= 1:
        flicker_rate = 0.0
    else:
        changes = sum(1 for a, b in zip(letters_only, letters_only[1:]) if a != b)
        flicker_rate = round(changes / (len(letters_only) - 1), 4)

    changes_per_second = round(flicker_rate * fps, 2)
    avg_conf = round(sum(c for _, c in non_none) / len(non_none), 3) if non_none else None

    return {
        "stable_pct": stable_pct,
        "flicker_rate": flicker_rate,
        "changes_per_second": changes_per_second,
        "avg_confidence": avg_conf,
    }


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(sequences=None, configs=None, fps=20.0, out_dir="eval_results", save=True):
    """
    Run the full smoothing ablation.

    Parameters
    ----------
    sequences : dict | None
        Maps sequence name -> list of (letter, confidence). Defaults to SEQUENCES.
    configs : list | None
        List of config dicts. Defaults to CONFIGS.
    fps : float
        Frames-per-second assumption used for changes_per_second metric.
    out_dir : str
        Directory to write the JSON results file.
    save : bool
        When False, skip writing the JSON file.

    Returns
    -------
    dict
        Full results dict including timestamp, fps_assumption, note, and per-sequence metrics.
    """
    if sequences is None:
        sequences = SEQUENCES
    if configs is None:
        configs = CONFIGS

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results = {
        "timestamp": timestamp,
        "fps_assumption": fps,
        "note": (
            "window<=7 all require min_majority=3 (max(3, window*0.4)); "
            "window=3 requires unanimous agreement"
        ),
        "sequences": {},
    }

    for seq_name, sequence in sequences.items():
        n_frames = len(sequence)
        results["sequences"][seq_name] = {}

        # Print table header
        print(f"\nSequence: {seq_name} ({n_frames} frames)")
        header = f"{'Config':<24} | {'stable_pct':>10} | {'flicker_rate':>12} | {'chg/sec':>7} | {'avg_conf':>9}"
        separator = "-" * 24 + "-+-" + "-" * 10 + "-+-" + "-" * 12 + "-+-" + "-" * 7 + "-+-" + "-" * 9
        print(header)
        print(separator)

        for cfg in configs:
            cfg_name = cfg["name"]
            cfg_type = cfg["type"]

            if cfg_type == "no_smooth":
                outputs = _no_smooth(sequence)
            elif cfg_type == "smooth":
                outputs = _smooth(sequence, window_size=cfg["window"])
            else:
                raise ValueError(f"Unknown config type: {cfg_type!r}")

            metrics = _compute_metrics(outputs, fps=fps)
            results["sequences"][seq_name][cfg_name] = metrics

            avg_conf_display = f"{metrics['avg_confidence']:.3f}" if metrics["avg_confidence"] is not None else "  N/A"
            print(
                f"{cfg_name:<24} | "
                f"{metrics['stable_pct']:>9.1f}% | "
                f"{metrics['flicker_rate']:>12.4f} | "
                f"{metrics['changes_per_second']:>7.2f} | "
                f"{avg_conf_display:>9}"
            )

    # Save results
    save_path = None
    if save:
        os.makedirs(out_dir, exist_ok=True)
        filename = f"smoothing_ablation_{timestamp}.json"
        save_path = os.path.join(out_dir, filename)
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    return results, save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate temporal smoothing window sizes on sign language prediction stability."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Skip saving results to disk (default: save).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frames-per-second assumption for changes_per_second metric (default: 20.0).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="eval_results",
        help="Directory to write JSON results (default: eval_results).",
    )
    args = parser.parse_args()

    results, save_path = run_ablation(
        fps=args.fps,
        out_dir=args.out_dir,
        save=not args.no_save,
    )

    print()
    if save_path:
        print(f"Results saved to: {save_path}")
    else:
        print("Saving skipped (--no-save flag set).")
