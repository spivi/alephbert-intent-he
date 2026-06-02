"""Shared helpers for baseline scripts.

Each baseline produces a JSON file with the same schema as `evaluate.py`'s
output, so `generate_evaluation_report.py` and the comparison renderer can
consume them uniformly:

    {
      "name": "<baseline name>",
      "accuracy": float,
      "mean_confidence": float,    # 0.0 if the baseline has no confidence notion
      "per_intent": classification_report dict,
      "confusion_matrix": list[list[int]],
      "num_samples": int,
      "extras": dict,              # baseline-specific notes (cost, latency, etc.)
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    """Return (texts, labels)."""
    texts: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def write_metrics(
    name: str,
    true_labels: list[int],
    pred_labels: list[int],
    confidences: list[float],
    label_names: list[str],
    output_path: Path,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the standard metrics and write a JSON report."""
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        labels=list(range(len(label_names))),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        true_labels, pred_labels, labels=list(range(len(label_names)))
    )
    payload = {
        "name": name,
        "accuracy": float(accuracy),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "per_intent": report,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(true_labels),
        "extras": extras or {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "[%s] accuracy=%.4f weighted-F1=%.4f macro-F1=%.4f → %s",
        name,
        accuracy,
        report["weighted avg"]["f1-score"],
        report["macro avg"]["f1-score"],
        output_path,
    )
    return payload
