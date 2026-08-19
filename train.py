#!/usr/bin/env python3
"""Train the frozen-Whisper cached-embedding GRU baseline with resumable checkpoints."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from turn_detection.datasets import CachedSequenceDataset, collate_cached
from turn_detection.metrics import by_source
from turn_detection.model import GRUTurnClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-base)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - p_t).pow(self.gamma) * base).mean()


def valid_loss(logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    mask = labels >= 0
    return criterion(logits[mask], labels[mask])


def evaluate(model, loader, device, criterion) -> tuple[float, list[dict]]:
    model.eval(); loss_sum = 0.0; batches = 0; records = []
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch["embeddings"].to(device), batch["lengths"].to(device))
            labels = batch["labels"].to(device)
            loss_sum += valid_loss(logits, labels, criterion).item(); batches += 1
            probabilities = torch.sigmoid(logits).cpu()
            for i, length in enumerate(batch["lengths"].tolist()):
                records.append({"id": batch["ids"][i], "source": batch["sources"][i], "labels": labels[i, :length].cpu().tolist(), "probabilities": probabilities[i, :length].tolist()})
    return loss_sum / max(batches, 1), records


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint(path, epoch, batch_in_epoch, epoch_indices, epoch_complete, global_step, model, optimizer, best_f1, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "cached_gru", "epoch": epoch, "batch_in_epoch": batch_in_epoch, "epoch_indices": epoch_indices,
                "epoch_complete": epoch_complete, "global_step": global_step, "best_f1": best_f1, "rng_state": rng_state(),
                "model_config": {"input_size": 384, "hidden_size": args.hidden_size, "num_layers": args.num_layers, "dropout": args.dropout},
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "args": vars(args)}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["weighted_bce", "focal"], default="weighted_bce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0, help="Save last.pt every N optimizer steps; enables exact mid-epoch resume.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Stop after N total optimizer steps, retaining a resumable partial-epoch checkpoint.")
    parser.add_argument("--step-trace", type=Path, default=None, help="Append JSONL batch identities after each optimizer step; useful for resume verification.")
    args = parser.parse_args(); seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CachedSequenceDataset(args.cache_dir / "train.jsonl", ROOT)
    val_ds = CachedSequenceDataset(args.cache_dir / "val.jsonl", ROOT)
    test_ds = CachedSequenceDataset(args.cache_dir / "test.jsonl", ROOT)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=args.num_workers, collate_fn=collate_cached)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=args.num_workers, collate_fn=collate_cached)
    positives = sum(int(item["labels"].sum()) for item in train_ds)
    total = sum(len(item["labels"]) for item in train_ds)
    pos_weight = torch.tensor([(total - positives) / max(positives, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if args.loss == "weighted_bce" else FocalLoss(alpha=float(pos_weight.item() / (1 + pos_weight.item())))
    model = GRUTurnClassifier(hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-2)
    start_epoch = 0; resume_batch = 0; resume_indices = None; global_step = 0; best_f1 = -1.0
    resume_path = args.resume or (args.checkpoint_dir / "last.pt")
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        global_step, best_f1 = state["global_step"], state["best_f1"]
        if "rng_state" in state:
            restore_rng(state["rng_state"])
        if state.get("epoch_complete", True):
            start_epoch = state["epoch"] + 1
        else:
            start_epoch, resume_batch, resume_indices = state["epoch"], state["batch_in_epoch"], state["epoch_indices"]
        print(f"Resuming {resume_path}: epoch={start_epoch}, next_batch={resume_batch}, global_step={global_step}.")
    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train(); running = 0.0
        epoch_indices = resume_indices if epoch == start_epoch and resume_indices is not None else torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [epoch_indices[index:index + args.batch_size] for index in range(0, len(epoch_indices), args.batch_size)]
        batch_start = resume_batch if epoch == start_epoch else 0
        train_loader = DataLoader(Subset(train_ds, sum(batches[batch_start:], [])), args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_cached, pin_memory=device.type == "cuda")
        for offset, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"), start=batch_start):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["embeddings"].to(device), batch["lengths"].to(device))
            loss = valid_loss(logits, batch["labels"].to(device), criterion)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            running += loss.item(); global_step += 1
            next_batch = offset + 1
            if args.step_trace:
                args.step_trace.parent.mkdir(parents=True, exist_ok=True)
                with args.step_trace.open("a") as handle:
                    handle.write(json.dumps({"global_step": global_step, "epoch": epoch,
                                             "completed_batch": next_batch, "ids": batch["ids"]}) + "\n")
            if args.checkpoint_every_steps and global_step % args.checkpoint_every_steps == 0:
                checkpoint(args.checkpoint_dir / "last.pt", epoch, next_batch, epoch_indices, False, global_step, model, optimizer, best_f1, args)
                print(f"checkpoint saved: epoch={epoch}, next_batch={next_batch}, global_step={global_step}")
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                checkpoint(args.checkpoint_dir / "last.pt", epoch, next_batch, epoch_indices, False, global_step, model, optimizer, best_f1, args)
                print(f"stopped intentionally: epoch={epoch}, next_batch={next_batch}, global_step={global_step}")
                return
        val_loss, records = evaluate(model, val_loader, device, criterion)
        metrics = by_source(records)
        macro_f1 = float(np.mean([m["f1"] for m in metrics.values()])) if metrics else 0.0
        history.append({"epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val_loss": val_loss, "metrics": metrics})
        print(json.dumps(history[-1], indent=2))
        if macro_f1 > best_f1:
            best_f1 = macro_f1; checkpoint(args.checkpoint_dir / "best.pt", epoch, len(batches), epoch_indices, True, global_step, model, optimizer, best_f1, args)
        checkpoint(args.checkpoint_dir / "last.pt", epoch, len(batches), epoch_indices, True, global_step, model, optimizer, best_f1, args)
        (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
    state = torch.load(args.checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    _, test_records = evaluate(model, test_loader, device, criterion)
    report = {"test": by_source(test_records), "checkpoint": str(args.checkpoint_dir / "best.pt")}
    (args.checkpoint_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    (args.checkpoint_dir / "test_window_predictions.json").write_text(json.dumps(test_records, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
