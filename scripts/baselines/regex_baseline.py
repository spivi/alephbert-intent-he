"""Hand-crafted keyword/regex baseline.

A useful "what would you build in 30 minutes without ML" reference. Falls back
to OTHER when no rule matches.

The rules below are not exhaustive. They are the kind of patterns a
Hebrew-speaking developer would write after looking at a few hundred sample
messages. If a fine-tuned model cannot beat this simple baseline, it is probably
not worth the trouble of training one.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _base import load_jsonl, write_metrics  # noqa: E402
from label_map import INTENT_LABELS, LABEL2ID  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Patterns are tried in order; first match wins. Tune-as-you-go list, not exhaustive.
_PATTERNS: list[tuple[str, str]] = [
    ("BUG_REPORT", r"^/bug\b|\bbug\b|לא עובד|שגיאה|תקלה"),
    ("RECIPE_URL", r"https?://\S+"),
    ("GET_INVITE_CODE", r"מה הקוד|תן קוד|תן לי קוד|הצג קוד|קוד שלי|ההזמנה שלי"),
    ("CREATE_INVITE", r"צור קוד|תייצר קוד|רוצה להזמין|הזמן|צור הזמנה"),
    ("REVOKE_INVITE", r"בטל קוד|מחק קוד|בטל הזמנה|תבטל קוד"),
    ("RENAME_GROUP", r"שנה שם|תשנה.*שם|שם חדש|תקרא לקבוצה|עדכן שם"),
    ("LEAVE_GROUP", r"עזוב|צא מהקבוצה|אני עוזב|לעזוב|תוציא אותי|הסר אותי"),
    ("GROUP_INFO", r"מי בקבוצה|פרטי קבוצה|חברי הקבוצה|מי נמצא|כמה אנשים"),
    ("NOTIFICATION_SETTINGS", r"התראות|הודעות|השתק|השתיק|כבה|הדלק|הפעל"),
    ("LIST_QUERY", r"מה ברשימה|הראה|תראה|הצג|רשימה|what.*list|show.*list"),
    ("CLEAR_LIST", r"סיימתי|קניתי הכל|נקה|הכל נקנה|גמרתי|clear"),
    ("REMOVE_ITEM", r"^(תמחק|תוריד|מחק|הסר|תסיר)\b"),
    ("PARTIAL_COMPLETION", r"חוץ מ|מלבד|פרט ל|כמעט הכל"),
    ("UPDATE_QUANTITY", r"בעצם|במקום|התכוונתי|תעדכן|כמות|במקום \d"),
    ("RECIPE_SEARCH", r"תכין.*רשימה|רשימה ל|מה צריך ל|מתכון ל"),
    ("GROCERY_REQUEST", r"^(תוסיף|צריך|תן לי|3|2|חצי קילו|קילו)\b|אני צריכ"),
]


def _classify(text: str) -> int:
    for intent, pattern in _PATTERNS:
        if re.search(pattern, text):
            return LABEL2ID[intent]
    return LABEL2ID["OTHER"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regex baseline")
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    texts, true_labels = load_jsonl(args.test_data)
    pred_labels = [_classify(t) for t in texts]

    # Confidence here is "rule matched" (1.0) vs "fell through to OTHER" (0.0)
    other_id = LABEL2ID["OTHER"]
    confidences = [
        0.0
        if (p == other_id and not any(re.search(p2, t) for _, p2 in _PATTERNS))
        else 1.0
        for t, p in zip(texts, pred_labels)
    ]

    write_metrics(
        name="regex",
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidences=confidences,
        label_names=INTENT_LABELS,
        output_path=args.output,
        extras={
            "description": f"{len(_PATTERNS)} hand-crafted Hebrew keyword/regex rules; falls back to OTHER",
            "num_rules": len(_PATTERNS),
            "cost_per_1k": "$0",
            "latency_ms_per_call": "< 0.1",
        },
    )


if __name__ == "__main__":
    main()
