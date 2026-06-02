"""Export fine-tuned AlephBERT to ONNX format.

Converts a PyTorch checkpoint to ONNX so it can run on CPU via
onnxruntime, without PyTorch installed. Validates that the output matches
PyTorch within tolerance.

Usage:
    python scripts/export_onnx.py \
        --model-dir models/alephbert-intent \
        --output models/alephbert-intent.onnx
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_TEST_SENTENCES = [
    "תוסיף חלב וביצים",
    "מה ברשימה?",
    "סיימתי קניות",
    "https://example.com/recipe",
    "שלום, מה נשמע?",
]


def _export_model(model_dir: str, onnx_path: str) -> None:
    """Export PyTorch model to ONNX."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy = tokenizer(
        "תוסיף חלב",
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )

    # dynamo=False forces the legacy TorchScript-based exporter.
    # The dynamo exporter in torch >= 2.10 silently drops weights when
    # exporting BertForSequenceClassification, producing a tiny ONNX file
    # whose accuracy collapses to near-random. The legacy exporter handles
    # dynamic_axes correctly and preserves the full model.
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    logger.info("Exported ONNX model to %s", onnx_path)


def _validate(model_dir: str, onnx_path: str) -> None:
    """Validate ONNX output matches PyTorch."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    pt_model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    for text in _TEST_SENTENCES:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        with torch.no_grad():
            pt_logits = pt_model(**inputs).logits.numpy()

        onnx_logits = session.run(
            None,
            {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
            },
        )[0]

        if not np.allclose(pt_logits, onnx_logits, atol=1e-4):
            raise ValueError(f"ONNX mismatch for: {text}")

    logger.info(
        "Validation passed: ONNX matches PyTorch for %d test sentences",
        len(_TEST_SENTENCES),
    )


def _copy_tokenizer(model_dir: str, tokenizer_dir: str) -> None:
    """Copy tokenizer files so the ONNX model runs without transformers."""
    out = Path(tokenizer_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(model_dir)

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, out / name)

    logger.info("Tokenizer files copied to %s", tokenizer_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AlephBERT to ONNX")
    parser.add_argument("--model-dir", default="models/alephbert-intent")
    parser.add_argument("--output", default="models/alephbert-intent.onnx")
    parser.add_argument("--tokenizer-dir", default="models/tokenizer")
    args = parser.parse_args()

    _export_model(args.model_dir, args.output)
    _validate(args.model_dir, args.output)
    _copy_tokenizer(args.model_dir, args.tokenizer_dir)

    logger.info("Done. Output files: %s, %s", args.output, args.tokenizer_dir)


if __name__ == "__main__":
    main()
