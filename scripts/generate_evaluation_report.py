"""Render an evaluation JSON report as Markdown + a confusion-matrix PNG.

Consumes the JSON produced by `evaluate.py` and emits:
  - EVALUATION.md         (per-intent precision/recall/F1 table + summary)
  - confusion_matrix.png  (normalized, color-coded)

Handy for the model card and the README. Pure formatting, no model loading.

Usage:
    python scripts/generate_evaluation_report.py \\
        --input  data/eval_report.json \\
        --md-out scripts/EVALUATION.md \\
        --png-out scripts/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from label_map import ID2LABEL, INTENT_LABELS, NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _format_md_table(report: dict[str, dict[str, float]]) -> str:
    """Render the per-intent classification report as a Markdown table."""
    lines = [
        "| Intent | Precision | Recall | F1 | Support |",
        "|--------|----------:|-------:|---:|--------:|",
    ]
    for intent in INTENT_LABELS:
        row = report.get(intent, {})
        if not row:
            continue
        lines.append(
            f"| `{intent}` "
            f"| {row.get('precision', 0):.3f} "
            f"| {row.get('recall', 0):.3f} "
            f"| {row.get('f1-score', 0):.3f} "
            f"| {int(row.get('support', 0))} |"
        )
    return "\n".join(lines)


def _format_summary(data: dict) -> str:
    """Render summary metrics (overall accuracy + macro/weighted averages)."""
    per = data["per_intent"]
    macro = per.get("macro avg", {})
    weighted = per.get("weighted avg", {})
    return (
        f"- **Accuracy:** {data['accuracy']:.4f}\n"
        f"- **Mean confidence:** {data['mean_confidence']:.4f}\n"
        f"- **Macro F1:** {macro.get('f1-score', 0):.4f}\n"
        f"- **Weighted F1:** {weighted.get('f1-score', 0):.4f}\n"
        f"- **Test samples:** {data['num_samples']}\n"
    )


def _write_markdown(data: dict, md_path: Path, png_filename: str) -> None:
    """Write EVALUATION.md."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f"# Evaluation Report: AlephBERT Hebrew Intent Classifier\n\n"
        f"_Generated: {timestamp}_\n\n"
        f"## Summary\n\n"
        f"{_format_summary(data)}\n"
        f"## Per-Intent Metrics\n\n"
        f"{_format_md_table(data['per_intent'])}\n\n"
        f"## Confusion Matrix\n\n"
        f"![Confusion matrix]({png_filename})\n\n"
        f"## Methodology\n\n"
        f"- Test split: held-out 20% of synthetic data "
        f"(`data/test.jsonl`)\n"
        f"- Model: ONNX inference on CPU\n"
        f"- Metrics computed via `scikit-learn.metrics.classification_report`\n"
        f"- Reproducible: run `python scripts/evaluate.py` "
        f"with the same seed\n"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(body, encoding="utf-8")
    logger.info("Wrote %s", md_path)


def _write_confusion_png(cm: list[list[int]], png_path: Path) -> None:
    """Render the confusion matrix as a row-normalized PNG."""
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_arr / row_sums

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    ax.set_xticks(np.arange(NUM_LABELS))
    ax.set_yticks(np.arange(NUM_LABELS))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    threshold = 0.5
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            value = cm_norm[i, j]
            if value < 0.01:
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=6,
            )

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", png_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render eval_report.json as Markdown + PNG"
    )
    parser.add_argument(
        "--input",
        default="data/eval_report.json",
        help="Path to evaluate.py JSON output",
    )
    parser.add_argument(
        "--md-out",
        default="scripts/EVALUATION.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--png-out",
        default="scripts/confusion_matrix.png",
        help="Confusion matrix PNG path",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    _write_confusion_png(data["confusion_matrix"], Path(args.png_out))
    _write_markdown(
        data,
        Path(args.md_out),
        png_filename=Path(args.png_out).name,
    )


if __name__ == "__main__":
    main()
