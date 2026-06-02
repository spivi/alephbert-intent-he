"""Run training N times with different seeds; aggregate metrics into mean ± std.

Each run trains, exports to ONNX, and evaluates on the SAME held-out test set,
varying ONLY the training seed (data split is fixed by `--data-dir`). Writes
one `eval_report_seed<N>.json` per run plus a consolidated `multi_run.json`
with per-intent F1 mean ± std across runs.

Usage:
    python scripts/multi_run_train.py \\
        --data-dir   data \\
        --output-dir runs/multi_run \\
        --n-runs 3 --base-seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path

# Flattened standalone-repo layout: all pipeline scripts are siblings here.
HF_CLASSIFIER_DIR = Path(__file__).resolve().parent
PUBLISH_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def _run_one(seed: int, data_dir: Path, output_dir: Path) -> dict:
    """Train + export ONNX + evaluate. Returns the eval JSON."""
    run_dir = output_dir / f"seed_{seed}"
    ckpt_dir = run_dir / "checkpoint"
    onnx_path = run_dir / "model.onnx"
    eval_path = run_dir / "eval.json"

    run_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            str(HF_CLASSIFIER_DIR / "train_classifier.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(ckpt_dir),
            "--seed",
            str(seed),
        ]
    )
    _run(
        [
            sys.executable,
            str(HF_CLASSIFIER_DIR / "export_onnx.py"),
            "--model-dir",
            str(ckpt_dir),
            "--output",
            str(onnx_path),
        ]
    )
    _run(
        [
            sys.executable,
            str(HF_CLASSIFIER_DIR / "evaluate.py"),
            "--onnx-model",
            str(onnx_path),
            "--tokenizer-dir",
            str(ckpt_dir),
            "--test-data",
            str(data_dir / "test.jsonl"),
            "--output",
            str(eval_path),
        ]
    )

    return json.loads(eval_path.read_text(encoding="utf-8"))


def _aggregate(reports: list[dict], n_runs: int) -> dict:
    """mean ± std for accuracy, F1s, and per-intent F1s."""
    accuracies = [r["accuracy"] for r in reports]
    weighted_f1s = [r["per_intent"]["weighted avg"]["f1-score"] for r in reports]
    macro_f1s = [r["per_intent"]["macro avg"]["f1-score"] for r in reports]

    intents = [
        k
        for k in reports[0]["per_intent"]
        if k not in {"accuracy", "macro avg", "weighted avg"}
    ]
    per_intent = {}
    for intent in intents:
        f1s = [r["per_intent"][intent]["f1-score"] for r in reports]
        supports = [r["per_intent"][intent]["support"] for r in reports]
        per_intent[intent] = {
            "f1_mean": statistics.mean(f1s),
            "f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "support_mean": statistics.mean(supports),
        }

    return {
        "n_runs": n_runs,
        "seeds": [42 + i for i in range(n_runs)],
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "accuracy_values": accuracies,
        "weighted_f1_mean": statistics.mean(weighted_f1s),
        "weighted_f1_std": statistics.stdev(weighted_f1s)
        if len(weighted_f1s) > 1
        else 0.0,
        "weighted_f1_values": weighted_f1s,
        "macro_f1_mean": statistics.mean(macro_f1s),
        "macro_f1_std": statistics.stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0,
        "macro_f1_values": macro_f1s,
        "per_intent": per_intent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-run training for variance estimation"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    for i in range(args.n_runs):
        seed = args.base_seed + i
        logger.info("=== Run %d / %d (seed=%d) ===", i + 1, args.n_runs, seed)
        reports.append(_run_one(seed, args.data_dir, args.output_dir))

    aggregate = _aggregate(reports, args.n_runs)
    summary_path = args.output_dir / "multi_run.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))

    logger.info("=== Summary (n=%d) ===", args.n_runs)
    logger.info(
        "accuracy:    %.4f ± %.4f  (values: %s)",
        aggregate["accuracy_mean"],
        aggregate["accuracy_std"],
        [f"{v:.4f}" for v in aggregate["accuracy_values"]],
    )
    logger.info(
        "weighted F1: %.4f ± %.4f",
        aggregate["weighted_f1_mean"],
        aggregate["weighted_f1_std"],
    )
    logger.info(
        "macro F1:    %.4f ± %.4f",
        aggregate["macro_f1_mean"],
        aggregate["macro_f1_std"],
    )
    logger.info("Wrote consolidated report to %s", summary_path)


if __name__ == "__main__":
    main()
