#!/usr/bin/env python3
"""Download/filter Pipecat data, merge recorded Hinglish clips, and split by clip."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "src"))
from turn_detection.audio import load_dataset_audio


def bool_or_false(value: object) -> bool:
    # Dataset booleans can be Arrow/Pandas null. Only a literal True is positive;
    # this avoids treating an unexpected non-empty string as a positive label.
    return value is True


def stratum(row: dict) -> str:
    # Filler presence is intentionally part of the split stratum.
    return f"{int(bool_or_false(row['endpoint_bool']))}_{int(bool_or_false(row.get('midfiller')))}_{int(bool_or_false(row.get('endfiller')))}"


def split_rows(rows: list[dict], seed: int, val_fraction: float, test_fraction: float) -> dict[str, list[dict]]:
    labels = [stratum(row) for row in rows]
    counts = Counter(labels)
    # Tiny hand-recorded subsets may not have two examples per stratum; keep them random but clip-level.
    stratify = labels if min(counts.values()) >= 2 else None
    indices = np.arange(len(rows))
    first = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction + test_fraction, random_state=seed)
    train_idx, held_idx = next(first.split(indices, stratify))
    held_labels = [labels[i] for i in held_idx]
    held_stratify = held_labels if min(Counter(held_labels).values()) >= 2 else None
    second = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction / (val_fraction + test_fraction), random_state=seed)
    val_local, test_local = next(second.split(held_idx, held_stratify))
    return {
        "train": [rows[i] for i in train_idx],
        "val": [rows[held_idx[i]] for i in val_local],
        "test": [rows[held_idx[i]] for i in test_local],
    }


def save_audio(example: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, load_dataset_audio(example["audio"]), 16_000)


def stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pipecat_rows(limit: int | None, raw_dir: Path, languages: set[str]) -> list[dict]:
    dataset = load_dataset("pipecat-ai/smart-turn-data-v3.2-train", split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    rows: list[dict] = []
    language_counts: Counter[str] = Counter()
    filler_counts: Counter[str] = Counter()
    for example in tqdm(dataset, desc="Filtering real Pipecat clips"):
        if bool_or_false(example.get("synthetic")):
            continue
        if str(example.get("language")) not in languages:
            continue
        language_counts[str(example["language"])] += 1
        filler_counts["midfiller"] += int(bool_or_false(example.get("midfiller")))
        filler_counts["endfiller"] += int(bool_or_false(example.get("endfiller")))
        path = raw_dir / "pipecat" / f"{example['id']}.wav"
        save_audio(example, path)
        rows.append({
            "id": str(example["id"]), "source": "pipecat", "audio_path": stored_path(path),
            "language": str(example["language"]), "endpoint_bool": bool_or_false(example["endpoint_bool"]),
            "midfiller": bool_or_false(example.get("midfiller")), "endfiller": bool_or_false(example.get("endfiller")),
            "synthetic": False, "dataset": str(example.get("dataset", "")), "transcript": example.get("spoken_text") or "",
        })
        if limit and len(rows) >= limit:
            break
    print("Real-only Pipecat rows:", len(rows))
    print("Language counts:", dict(language_counts))
    print("Filler counts:", dict(filler_counts))
    return rows


def hinglish_rows(manifest: Path) -> list[dict]:
    if not manifest.exists():
        return []
    frame = pd.read_csv(manifest)
    required = {"id", "audio_file", "endpoint_bool", "transcript"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest} is missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for item in frame.to_dict("records"):
        audio_path = ROOT / "data" / "hinglish_recordings" / str(item["audio_file"])
        if not audio_path.exists():
            continue
        rows.append({
            "id": f"hinglish-{item['id']}", "source": "hinglish", "audio_path": str(audio_path.relative_to(ROOT)),
            "language": "hinglish", "endpoint_bool": bool_or_false(item["endpoint_bool"]),
            "midfiller": bool_or_false(item.get("midfiller")), "endfiller": bool_or_false(item.get("endfiller")),
            "synthetic": False, "dataset": "hand_recorded_hinglish", "transcript": str(item["transcript"]),
        })
    print(f"Recorded Hinglish clips found: {len(rows)} / {len(frame)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw", help="Use Google Drive here in Colab so downloaded audio survives resets.")
    parser.add_argument("--hinglish-manifest", type=Path, default=ROOT / "data" / "hinglish_manifest.csv")
    parser.add_argument("--max-pipecat-clips", type=int, default=None, help="Use a small smoke-test subset before a Colab run.")
    parser.add_argument("--languages", nargs="+", default=["eng", "hin"], help="Retain only these Pipecat language codes (default: eng hin).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pipecat = pipecat_rows(args.max_pipecat_clips, args.raw_dir, set(args.languages))
    hinglish = hinglish_rows(args.hinglish_manifest)
    if len(pipecat) < 10:
        raise RuntimeError("Too few real Pipecat rows after filtering.")
    # Split each source separately so Hinglish is represented in validation and test once recordings exist.
    pipecat_splits = split_rows(pipecat, args.seed, 0.1, 0.1)
    hinglish_splits = split_rows(hinglish, args.seed, 0.1, 0.1) if len(hinglish) >= 10 else {k: [] for k in ("train", "val", "test")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = pipecat_splits[split] + hinglish_splits[split]
        with (args.output_dir / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(split, len(rows), Counter(row["source"] for row in rows))
    report = {
        "real_pipecat_clips": len(pipecat),
        "language_counts": dict(Counter(row["language"] for row in pipecat)),
        "endpoint_counts": dict(Counter(str(row["endpoint_bool"]) for row in pipecat)),
        "filler_counts": {"midfiller": sum(row["midfiller"] for row in pipecat), "endfiller": sum(row["endfiller"] for row in pipecat)},
        "split_counts": {split: len(pipecat_splits[split]) + len(hinglish_splits[split]) for split in ("train", "val", "test")},
        "hinglish_recorded_clips": len(hinglish),
    }
    (args.output_dir / "filter_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
