"""Majority-class baseline. Always predicts whichever label is most frequent in train.

This is a stronger floor than random for imbalanced datasets. For our (roughly
uniform) 17-class corpus it usually lands close to random.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _base import load_jsonl, write_metrics  # noqa: E402
from label_map import ID2LABEL, INTENT_LABELS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Majority-class baseline")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    _train_texts, train_labels = load_jsonl(args.train_data)
    most_common_label, count = Counter(train_labels).most_common(1)[0]
    logger.info(
        "Majority label in train: %s (id=%d, %d/%d)",
        ID2LABEL[most_common_label],
        most_common_label,
        count,
        len(train_labels),
    )

    _test_texts, true_labels = load_jsonl(args.test_data)
    pred_labels = [most_common_label] * len(true_labels)
    confidences = [count / len(train_labels)] * len(true_labels)

    write_metrics(
        name="majority",
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidences=confidences,
        label_names=INTENT_LABELS,
        output_path=args.output,
        extras={
            "description": f"Always predict {ID2LABEL[most_common_label]} (most-frequent train label)",
            "majority_label": ID2LABEL[most_common_label],
            "cost_per_1k": "$0",
            "latency_ms_per_call": 0,
        },
    )


if __name__ == "__main__":
    main()
