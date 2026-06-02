"""Evaluate ONNX intent classifier on test data.

Computes accuracy, per-intent precision/recall/F1, and confusion matrix.

Usage:
    python scripts/evaluate.py \
        --onnx-model models/alephbert-intent.onnx \
        --test-data data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tokenizers import Tokenizer

from label_map import ID2LABEL, NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_LENGTH = 128


def _load_test_data(path: str) -> tuple[list[str], list[int]]:
    """Load test JSONL and return (texts, labels)."""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def _predict(
    session: ort.InferenceSession,
    tokenizer: Tokenizer,
    texts: list[str],
    batch_size: int = 32,
) -> tuple[list[int], list[float]]:
    """Run ONNX inference in batches."""
    all_preds, all_confs = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer.encode_batch(batch)

        # tokenizer.json bakes in padding to MAX_LENGTH, so enc.ids is already
        # 128 long for every input. Use enc.attention_mask (computed by the
        # tokenizer itself). Naively setting attention_mask[:len(ids)] = 1
        # would attend to PAD tokens too and collapse accuracy to ~16%.
        input_ids = np.array(
            [enc.ids[:MAX_LENGTH] for enc in encodings], dtype=np.int64
        )
        attention_mask = np.array(
            [enc.attention_mask[:MAX_LENGTH] for enc in encodings], dtype=np.int64
        )

        logits = session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]

        probs = _softmax(logits)
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)

        all_preds.extend(preds.tolist())
        all_confs.extend(confs.tolist())

    return all_preds, all_confs


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def evaluate(
    onnx_path: str,
    tokenizer_path: str,
    test_path: str,
    output_path: str | None,
) -> dict:
    """Run full evaluation and return metrics."""
    texts, true_labels = _load_test_data(test_path)
    logger.info("Loaded %d test samples", len(texts))

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))

    pred_labels, confidences = _predict(session, tokenizer, texts)

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    accuracy = accuracy_score(true_labels, pred_labels)

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_LABELS)))

    print(f"\n{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean confidence: {np.mean(confidences):.4f}")
    print(f"{'=' * 60}")
    print(
        classification_report(
            true_labels,
            pred_labels,
            target_names=label_names,
            zero_division=0,
        )
    )

    result = {
        "accuracy": accuracy,
        "mean_confidence": float(np.mean(confidences)),
        "per_intent": report,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(texts),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
        logger.info("Report saved to %s", output_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONNX intent classifier")
    parser.add_argument("--onnx-model", default="models/alephbert-intent.onnx")
    parser.add_argument("--tokenizer-dir", default="models/tokenizer")
    parser.add_argument("--test-data", default="data/test.jsonl")
    parser.add_argument("--output", default="data/eval_report.json")
    args = parser.parse_args()

    evaluate(args.onnx_model, args.tokenizer_dir, args.test_data, args.output)


if __name__ == "__main__":
    main()
