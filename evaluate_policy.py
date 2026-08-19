#!/usr/bin/env python3
"""Analyse threshold/consecutive-frame policies from saved per-window predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from turn_detection.policy import ConsecutiveThresholdPolicy


def evaluate(records, threshold, consecutive):
    false_interruptions = 0; latencies = []
    for row in records:
        trigger = ConsecutiveThresholdPolicy(threshold, consecutive).first_trigger(row["probabilities"])
        positive_indices = [i for i, value in enumerate(row["labels"]) if value == 1]
        if not positive_indices:
            false_interruptions += int(trigger is not None)
        elif trigger is not None:
            latencies.append(max(0, trigger - positive_indices[-1]) * 0.2)
    return {"threshold": threshold, "consecutive_frames": consecutive, "false_interruptions": false_interruptions,
            "avg_detection_latency_seconds": sum(latencies) / len(latencies) if latencies else None, "detected_endpoints": len(latencies)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="JSON list emitted by a custom evaluation run.")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "policy_tradeoff.png")
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text())
    results = [evaluate(records, threshold, frames) for threshold in (0.5, 0.7, 0.9) for frames in (1, 2, 3)]
    print(json.dumps(results, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in results:
        if item["avg_detection_latency_seconds"] is not None:
            plt.scatter(item["false_interruptions"], item["avg_detection_latency_seconds"], label=f"p>{item['threshold']}, k={item['consecutive_frames']}")
    plt.xlabel("False interruptions (non-end clips)"); plt.ylabel("Average detection latency (s)"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(args.output, dpi=160)


if __name__ == "__main__": main()
