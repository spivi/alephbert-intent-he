"""Bake id2label / label2id into a checkpoint's config.json for HF Hub publish.

A HuggingFace `pipeline("text-classification", ...)` returns labels like
"LABEL_0" when a checkpoint's config.json lacks `id2label`. This script
takes a trained checkpoint directory and emits a publication-ready
directory where the config explicitly maps every numeric label to its
named intent, so downstream users get `"GROCERY_REQUEST"` out of the box.

Usage:
    python scripts/bake_id2label.py \\
        --input  models/alephbert-intent-ckpt \\
        --output models/alephbert-intent-publish

    # Optional self-test (loads the model + tokenizer end-to-end):
    python scripts/bake_id2label.py \\
        --input  models/alephbert-intent-ckpt \\
        --output models/alephbert-intent-publish \\
        --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from label_map import ID2LABEL, INTENT_LABELS, LABEL2ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_TEST_LABEL = "GROCERY_REQUEST"
EXPECTED_TEST_INPUT = "תוסיף חלב וביצים"


def _copy_checkpoint(input_dir: Path, output_dir: Path) -> None:
    """Mirror every file from input to output (model weights, tokenizer)."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)
    logger.info("Copied checkpoint → %s", output_dir)


def _bake_config(output_dir: Path, base_model: str) -> None:
    """Inject id2label + label2id + base_model into config.json."""
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing in {output_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    config["id2label"] = {str(i): label for i, label in ID2LABEL.items()}
    config["label2id"] = dict(LABEL2ID)

    config.setdefault("_name_or_path", base_model)
    config.setdefault("base_model", base_model)

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Baked id2label (%d labels) into %s", len(ID2LABEL), config_path)


def _self_test(output_dir: Path) -> None:
    """Load the baked model + run a Hebrew sample. Fail loud on LABEL_N."""
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model=str(output_dir),
        tokenizer=str(output_dir),
    )
    result = classifier(EXPECTED_TEST_INPUT)
    label = result[0]["label"] if isinstance(result, list) else result["label"]

    if label.startswith("LABEL_"):
        raise SystemExit(
            f"Self-test FAILED: pipeline returned generic '{label}'. "
            "id2label was not picked up by transformers."
        )

    if label not in INTENT_LABELS:
        raise SystemExit(
            f"Self-test FAILED: '{label}' is not in INTENT_LABELS. "
            "label_map.py and the checkpoint are out of sync."
        )

    logger.info(
        "Self-test PASSED: '%s' → '%s' (named label, not LABEL_N).",
        EXPECTED_TEST_INPUT,
        label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake id2label/label2id into a checkpoint's config.json"
    )
    parser.add_argument("--input", required=True, help="Source checkpoint dir")
    parser.add_argument("--output", required=True, help="Destination publish dir")
    parser.add_argument(
        "--base-model",
        default="onlplab/alephbert-base",
        help="Upstream base model identifier",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run pipeline() to verify named-label output (requires torch)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        logger.error("Input dir not found: %s", input_dir)
        sys.exit(1)

    _copy_checkpoint(input_dir, output_dir)
    _bake_config(output_dir, args.base_model)

    if args.self_test:
        _self_test(output_dir)


if __name__ == "__main__":
    main()
