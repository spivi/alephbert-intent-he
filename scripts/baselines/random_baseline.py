"""Uniform-random baseline. Predicts each of the 17 intents with equal probability.

This is the sanity floor: any classifier worth using must beat ~5.9% accuracy.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

# Ensure sibling label_map and _base import resolve when run from project root
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _base import load_jsonl, write_metrics  # noqa: E402
from label_map import INTENT_LABELS, NUM_LABELS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random baseline")
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    _texts, true_labels = load_jsonl(args.test_data)
    pred_labels = [rng.randrange(NUM_LABELS) for _ in true_labels]
    confidences = [1.0 / NUM_LABELS] * len(true_labels)

    write_metrics(
        name="random",
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidences=confidences,
        label_names=INTENT_LABELS,
        output_path=args.output,
        extras={
            "description": "Uniform random over 17 classes (sanity floor)",
            "cost_per_1k": "$0",
            "latency_ms_per_call": 0,
        },
    )


if __name__ == "__main__":
    main()
