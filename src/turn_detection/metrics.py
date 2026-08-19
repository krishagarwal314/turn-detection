from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def binary_metrics(labels: list[float], probabilities: list[float], threshold: float = 0.5) -> dict:
    y_true = np.asarray(labels, dtype=int)
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "support": int(y_true.sum()), "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def by_source(records: list[dict], threshold: float = 0.5) -> dict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"labels": [], "probabilities": []})
    for record in records:
        grouped[record["source"]]["labels"].extend(record["labels"])
        grouped[record["source"]]["probabilities"].extend(record["probabilities"])
    return {source: binary_metrics(values["labels"], values["probabilities"], threshold) for source, values in grouped.items()}
