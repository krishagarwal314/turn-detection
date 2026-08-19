from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CachedSequenceDataset(Dataset):
    def __init__(self, manifest: str | Path, root: str | Path):
        self.root = Path(root)
        with Path(manifest).open() as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        item = torch.load(self.root / row["cache_path"], map_location="cpu", weights_only=False)
        return {"embeddings": item["embeddings"].float(), "labels": item["labels"].float(), "source": row["source"], "id": row["id"]}


def collate_cached(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(item["labels"]) for item in batch], dtype=torch.long)
    return {
        "embeddings": pad_sequence([item["embeddings"] for item in batch], batch_first=True),
        "labels": pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-1.0),
        "lengths": lengths,
        "sources": [item["source"] for item in batch],
        "ids": [item["id"] for item in batch],
    }
