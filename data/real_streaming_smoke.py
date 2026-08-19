#!/usr/bin/env python3
"""Bounded real-data validation for the full frozen baseline pipeline.

This intentionally uses the Hugging Face streaming loader so it never materializes
the 41 GB dataset. It writes a self-contained smoke subset, including decoded audio.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from turn_detection.audio import labels_for_clip, load_dataset_audio, windows_for_audio


def as_bool(value: object) -> bool:
    # Pandas/Arrow null must be treated as absent, never as a truthy string.
    return value is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".validation_artifacts" / "real_smoke")
    parser.add_argument("--real-rows", type=int, default=300)
    parser.add_argument("--languages", nargs="+", default=["eng", "hin"], help="Retain only these language codes (default: eng hin).")
    args = parser.parse_args()
    raw_dir = args.output_dir / "raw"
    rows = []
    null_example = None
    short_example = None
    language_counts: Counter[str] = Counter()
    endpoint_counts: Counter[bool] = Counter()
    filler_counts: Counter[str] = Counter()
    stream = load_dataset("pipecat-ai/smart-turn-data-v3.2-train", split="train", streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))
    scanned = 0
    for example in stream:
        scanned += 1
        if example.get("synthetic") is not False or str(example.get("language")) not in set(args.languages):
            continue
        audio_array = load_dataset_audio(example["audio"])
        audio_path = raw_dir / f"{example['id']}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, audio_array, 16_000)
        row = {
            "id": str(example["id"]), "source": "pipecat", "audio_path": str(audio_path), "language": str(example["language"]),
            "endpoint_bool": as_bool(example.get("endpoint_bool")), "midfiller": as_bool(example.get("midfiller")),
            "endfiller": as_bool(example.get("endfiller")), "synthetic": False, "dataset": str(example.get("dataset", "")),
            "transcript": example.get("spoken_text") or "",
        }
        rows.append(row); language_counts[row["language"]] += 1; endpoint_counts[row["endpoint_bool"]] += 1
        filler_counts["midfiller_true"] += int(row["midfiller"]); filler_counts["endfiller_true"] += int(row["endfiller"])
        if null_example is None and (example.get("midfiller") is None or example.get("endfiller") is None):
            null_example = {"id": row["id"], "raw_midfiller": example.get("midfiller"), "raw_endfiller": example.get("endfiller"),
                            "stored_midfiller": row["midfiller"], "stored_endfiller": row["endfiller"]}
        if short_example is None and len(audio_array) < 16_000:
            windows = windows_for_audio(audio_array)
            short_example = {"id": row["id"], "duration_seconds": len(audio_array) / 16_000,
                             "sample_count": len(audio_array), "window_count": len(windows), "window_samples": len(windows[0]),
                             "zero_padding_samples": int((windows[0] == 0).sum()), "labels_if_endpoint": labels_for_clip(audio_array, True).tolist(),
                             "labels_if_not_endpoint": labels_for_clip(audio_array, False).tolist()}
        if len(rows) >= args.real_rows:
            break
    if len(rows) != args.real_rows:
        raise RuntimeError(f"Only collected {len(rows)} real rows after scanning {scanned} rows")
    # The baseline cache/training scripts require all three manifests. Deterministic clip-level split avoids window leakage.
    split_at_1, split_at_2 = int(len(rows) * 0.8), int(len(rows) * 0.9)
    splits = {"train": rows[:split_at_1], "val": rows[split_at_1:split_at_2], "test": rows[split_at_2:]}
    manifests = args.output_dir / "processed"; manifests.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        with (manifests / f"{split}.jsonl").open("w") as handle:
            for row in split_rows:
                handle.write(json.dumps(row) + "\n")
    report = {"loader": "load_dataset(..., streaming=True)", "scanned_rows": scanned, "real_rows": len(rows), "split_counts": {key: len(value) for key, value in splits.items()},
              "language_counts": dict(language_counts), "endpoint_counts": {str(k): v for k, v in endpoint_counts.items()}, "filler_counts": dict(filler_counts),
              "null_example": null_example, "short_clip_example": short_example}
    (args.output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
