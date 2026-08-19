from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from .audio import labels_for_clip, load_mono_audio, windows_for_audio
from .constants import WHISPER_MODEL_ID


class AudioSequenceDataset(Dataset):
    """Creates fresh one-second Whisper inputs; intentionally no cross-window encoder cache."""

    def __init__(self, manifest: str | Path, root: str | Path):
        self.root = Path(root)
        with Path(manifest).open() as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = load_mono_audio(self.root / row["audio_path"])
        features = self.processor(windows_for_audio(audio), sampling_rate=16_000, return_tensors="pt").input_features
        return {"features": features, "labels": torch.from_numpy(labels_for_clip(audio, row["endpoint_bool"])), "source": row["source"], "id": row["id"]}


def collate_audio(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(item["labels"]) for item in batch], dtype=torch.long)
    # All feature tensors have the same mel/time dimensions, only window count varies.
    return {
        "features": pad_sequence([item["features"] for item in batch], batch_first=True),
        "labels": pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-1.0),
        "lengths": lengths, "sources": [item["source"] for item in batch], "ids": [item["id"] for item in batch],
    }
