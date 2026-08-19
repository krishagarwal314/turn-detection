#!/usr/bin/env python3
"""Cache independent Whisper Tiny mean-pooled embeddings for each clip sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turn_detection.audio import labels_for_clip, load_mono_audio, windows_for_audio
from turn_detection.constants import WHISPER_MODEL_ID

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    with path.open() as handle:
        yield from (json.loads(line) for line in handle if line.strip())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    encoder = WhisperModel.from_pretrained(WHISPER_MODEL_ID).encoder.to(args.device).eval()
    for split in ("train", "val", "test"):
        output_manifest = []
        for row in tqdm(list(read_jsonl(args.manifest_dir / f"{split}.jsonl")), desc=f"Caching {split}"):
            cache_path = args.cache_dir / split / f"{row['id']}.pt"
            if not cache_path.exists() or args.overwrite:
                audio = load_mono_audio(ROOT / row["audio_path"])
                windows = windows_for_audio(audio)
                chunks = []
                for index in range(0, len(windows), args.batch_size):
                    features = processor(windows[index:index + args.batch_size], sampling_rate=16_000, return_tensors="pt").input_features
                    chunks.append(encoder(features.to(args.device)).last_hidden_state.mean(dim=1).cpu())
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"embeddings": torch.cat(chunks), "labels": torch.from_numpy(labels_for_clip(audio, row["endpoint_bool"])), "meta": row}, cache_path)
            # Cache may live on Google Drive, outside the cloned repository.
            # ``Path / absolute_path`` remains absolute when the dataset reloads it.
            output_manifest.append({**row, "cache_path": str(cache_path)})
        with (args.cache_dir / f"{split}.jsonl").open("w") as handle:
            for row in output_manifest:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
