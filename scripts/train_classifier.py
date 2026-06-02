"""Fine-tune AlephBERT for intent classification.

Trains onlplab/alephbert-base on labeled Hebrew grocery messages
using HuggingFace Trainer API with early stopping.

Usage:
    python scripts/train_classifier.py \
        --data-dir data \
        --output-dir models/alephbert-intent
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from label_map import ID2LABEL, LABEL2ID, NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "onlplab/alephbert-base"
MAX_LENGTH = 128


def _seed_all(seed: int) -> None:
    """Pin Python / NumPy / PyTorch RNGs.

    MPS (Apple Silicon) is not fully deterministic, so a minor variance of
    less than 0.5pp across runs with the same seed is expected. CPU and CUDA
    back-ends are
    deterministic given this seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize_dataset(
    data_dir: str,
    tokenizer_name: str,
) -> tuple[object, object, object]:
    """Load JSONL dataset and tokenize."""
    ds = load_dataset(
        "json",
        data_files={
            "train": str(Path(data_dir) / "train.jsonl"),
            "test": str(Path(data_dir) / "test.jsonl"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _tok(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

    tokenized = ds.map(_tok, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    return tokenized["train"], tokenized["test"], tokenizer


def _compute_metrics(eval_pred: object) -> dict[str, float]:
    """Compute accuracy and weighted F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train(
    model_name: str,
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    """Fine-tune model and return final metrics."""
    _seed_all(seed)

    train_ds, eval_ds, tokenizer = _tokenize_dataset(data_dir, model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save best model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label map
    label_map_path = Path(output_dir) / "label_map.json"
    label_map_path.write_text(json.dumps(LABEL2ID, ensure_ascii=False, indent=2))

    # Evaluate and save metrics
    metrics = trainer.evaluate()
    metrics_path = Path(output_dir) / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info(
        "Training complete. Accuracy: %.4f, F1: %.4f",
        metrics.get("eval_accuracy", 0),
        metrics.get("eval_f1_weighted", 0),
    )
    logger.info("Model saved to %s", output_dir)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune AlephBERT for intent classification"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument(
        "--data-dir", default="data", help="Training data dir"
    )
    parser.add_argument(
        "--output-dir", default="models/alephbert-intent", help="Output dir"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed pinned for Python/NumPy/PyTorch and Trainer",
    )
    args = parser.parse_args()

    train(
        args.model,
        args.data_dir,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
