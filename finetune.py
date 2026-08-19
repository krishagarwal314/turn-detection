#!/usr/bin/env python3
"""Optional low-LR experiment: unfreeze the final Whisper encoder layer(s) and train end-to-end."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from turn_detection.audio import labels_for_clip, load_mono_audio
from turn_detection.finetune_data import AudioSequenceDataset, collate_audio
from turn_detection.metrics import by_source
from turn_detection.model import WhisperGRUTurnClassifier


def loss_for(logits, labels, criterion):
    mask = labels >= 0
    return criterion(logits[mask], labels[mask])


def evaluate(model, loader, device, criterion):
    model.eval(); records = []; total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch["features"].to(device), batch["lengths"].to(device))
            labels = batch["labels"].to(device); total_loss += loss_for(logits, labels, criterion).item()
            probs = logits.sigmoid().cpu()
            for i, length in enumerate(batch["lengths"].tolist()):
                records.append({"id": batch["ids"][i], "source": batch["sources"][i], "labels": labels[i, :length].cpu().tolist(), "probabilities": probs[i, :length].tolist()})
    return total_loss / max(len(loader), 1), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "finetune")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2, help="Whisper feature tensors are large; Colab GPU memory is the limiting factor.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {split: AudioSequenceDataset(args.manifest_dir / f"{split}.jsonl", ROOT) for split in ("train", "val", "test")}
    loaders = {split: DataLoader(ds, args.batch_size, shuffle=split == "train", num_workers=args.num_workers, collate_fn=collate_audio) for split, ds in datasets.items()}
    model = WhisperGRUTurnClassifier(hidden_size=args.hidden_size, unfreeze_last_layers=args.unfreeze_last_layers).to(device)
    # Class ratio is from clip labels: each endpoint clip contributes one positive window.
    positives = sum(int(item["endpoint_bool"]) for item in datasets["train"].rows)
    total_windows = sum(len(labels_for_clip(load_mono_audio(Path(row["audio_path"]) if Path(row["audio_path"]).is_absolute() else ROOT / row["audio_path"]), row["endpoint_bool"])) for row in datasets["train"].rows)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(total_windows - positives, 1) / max(positives, 1)], device=device))
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-2)
    start = 0; best = -1.0; global_step = 0
    resume = args.resume or args.checkpoint_dir / "last.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        start, best, global_step = state["epoch"] + 1, state["best_f1"], state["global_step"]
        print(f"Resuming {resume} at epoch {start}")
    for epoch in range(start, args.epochs):
        model.train()
        for batch in tqdm(loaders["train"], desc=f"fine-tune epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["features"].to(device), batch["lengths"].to(device))
            loss = loss_for(logits, batch["labels"].to(device), criterion)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); global_step += 1
        val_loss, records = evaluate(model, loaders["val"], device, criterion); metrics = by_source(records)
        score = sum(item["f1"] for item in metrics.values()) / max(len(metrics), 1)
        state = {"model_type": "finetuned_whisper_gru", "epoch": epoch, "global_step": global_step, "best_f1": max(best, score),
                 "model_config": {"hidden_size": args.hidden_size, "unfreeze_last_layers": args.unfreeze_last_layers}, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_loss": val_loss, "metrics": metrics}
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True); torch.save(state, args.checkpoint_dir / "last.pt")
        if score > best: best = score; torch.save(state, args.checkpoint_dir / "best.pt")
        print(json.dumps({"epoch": epoch, "val_loss": val_loss, "metrics": metrics}, indent=2))
    best_state = torch.load(args.checkpoint_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(best_state["model_state"])
    _, records = evaluate(model, loaders["test"], device, criterion)
    report = by_source(records); (args.checkpoint_dir / "test_metrics.json").write_text(json.dumps(report, indent=2)); (args.checkpoint_dir / "test_window_predictions.json").write_text(json.dumps(records, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
