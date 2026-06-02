"""GPT-4o-mini zero-shot classifier baseline.

This is the comparison that matters: a tuned prompt sent to gpt-4o-mini, with no
fine-tuning. It shows what a general-purpose LLM gives you out of the box, so you
can judge whether a small fine-tuned model is worth training for the task.

Cost estimate: about $0.02 for the held-out test set (374 samples, roughly 250
tokens per prompt). Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _base import load_jsonl, write_metrics  # noqa: E402
from label_map import EN_DESCRIPTIONS, INTENT_LABELS, LABEL2ID  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_INTENT_BLOCK = "\n".join(
    f"- {intent}: {EN_DESCRIPTIONS[intent]}" for intent in INTENT_LABELS
)

_SYSTEM_PROMPT = (
    "You are a Hebrew text classifier for a shopping/grocery bot. "
    "Given a Hebrew message, return EXACTLY ONE intent label from the list below "
    "and nothing else. The label must be one of these exact strings (no quotes, no extra text):\n\n"
    f"{_INTENT_BLOCK}\n\n"
    "If the message does not match any intent (e.g. greetings, jokes, random text), "
    "return OTHER. Output ONLY the label string."
)


_VALID_LABELS = set(INTENT_LABELS)


async def _classify_one(
    client: httpx.AsyncClient,
    api_key: str,
    text: str,
    model: str,
) -> tuple[int, float]:
    """Return (predicted_label_id, confidence in [0, 1])."""
    try:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 20,
                "logprobs": True,
                "top_logprobs": 3,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.strip("`\"' \n")  # tolerate stray quotes / backticks

        # Sanity-fix: model sometimes adds trailing punctuation or whitespace
        for valid in _VALID_LABELS:
            if raw == valid:
                label_id = LABEL2ID[valid]
                # First-token logprob as a rough confidence proxy
                try:
                    lp = data["choices"][0]["logprobs"]["content"][0]["logprob"]
                    import math

                    conf = math.exp(lp)
                except Exception:
                    conf = 1.0
                return label_id, float(conf)

        logger.debug("Unparseable GPT output %r → OTHER", raw)
        return LABEL2ID["OTHER"], 0.0
    except Exception:
        logger.exception("GPT call failed for: %s", text[:60])
        return LABEL2ID["OTHER"], 0.0


async def classify_all(
    texts: list[str], api_key: str, model: str, concurrency: int
) -> tuple[list[int], list[float]]:
    sem = asyncio.Semaphore(concurrency)
    pred_labels: list[int] = [LABEL2ID["OTHER"]] * len(texts)
    confidences: list[float] = [0.0] * len(texts)

    async with httpx.AsyncClient() as client:

        async def _wrapped(i: int, t: str) -> None:
            async with sem:
                pred_labels[i], confidences[i] = await _classify_one(
                    client, api_key, t, model
                )

        await asyncio.gather(*[_wrapped(i, t) for i, t in enumerate(texts)])

    return pred_labels, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-4o-mini zero-shot baseline")
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "test_openai_key":
        logger.error("OPENAI_API_KEY not set")
        raise SystemExit(1)

    texts, true_labels = load_jsonl(args.test_data)
    logger.info(
        "Classifying %d samples via %s (concurrency=%d)",
        len(texts),
        args.model,
        args.concurrency,
    )

    t0 = time.monotonic()
    pred_labels, confidences = asyncio.run(
        classify_all(texts, api_key, args.model, args.concurrency)
    )
    elapsed = time.monotonic() - t0

    write_metrics(
        name=f"{args.model}-zeroshot",
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidences=confidences,
        label_names=INTENT_LABELS,
        output_path=args.output,
        extras={
            "description": f"Structured prompt → {args.model}, temperature=0, no examples",
            "model": args.model,
            "concurrency": args.concurrency,
            "wall_time_s": round(elapsed, 1),
            "latency_ms_per_call": round(elapsed * 1000 / len(texts), 1),
            "cost_per_1k": "~$0.05 (gpt-4o-mini Jan 2026 pricing, ~250-token prompt)",
        },
    )


if __name__ == "__main__":
    main()
