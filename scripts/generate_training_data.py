"""Generate training data via teacher-student distillation.

Uses GPT-4o-mini to paraphrase Hebrew seed templates into diverse
training examples for the AlephBERT intent classifier.

Usage:
    python scripts/generate_training_data.py \
        --output-dir data \
        --variations 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path

import httpx

from label_map import LABEL2ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---- Seed templates per intent (Hebrew grocery bot context) ----

SEED_TEMPLATES: dict[str, list[str]] = {
    "GROCERY_REQUEST": [
        "תוסיף חלב וביצים",
        "צריך 2 קילו עגבניות",
        "תן לי לחם, חמאה ו3 יוגורטים",
        "חלב",
        "שתי חבילות פסטה ורוטב עגבניות",
        "🛒 תוסיף בננות ותפוחים",
        "add milk and לחם",
        "חאלב וביצים בבקשה",
        "תוסיף שקיות אשפה ונייר טואלט",
        "אני צריכה גבינה צהובה, גבינה לבנה, ושמנת",
        "תוסיף קילו בצל וחצי קילו גזר",
        "3 קרטון ביצים",
        "במבה, ביסלי, ושוקולד",
        "קפה, סוכר, חלב, תה",
        "תוסיף עוף שלם ו2 חזה עוף",
        "פנקו, שמן, קמח",
        "שמיר ופטרוזיליה",
        "תוסיף סבון כלים ואקונומיקה",
        "5 בננות ו3 תפוזים",
        "חצי קילו גבינה צהובה",
    ],
    "RECIPE_URL": [
        "https://www.hashulchan.co.il/recipe/shakshuka",
        "https://foodish.co.il/recipe/12345",
        "הנה מתכון: https://example.com/recipe",
        "https://www.10dakot.co.il/pasta-recipe",
        "תבדוק את זה https://recipes.co.il/cake",
        "http://food.walla.co.il/recipe/123",
        "https://www.al-hashulchan.co.il/חומוס",
        "https://youtube.com/watch?v=recipe123",
        "מתכון לעוגה https://myblog.com/cake",
        "https://www.lecker.co.il/מתכון-פשטידה",
    ],
    "LIST_QUERY": [
        "מה ברשימה?",
        "הצג רשימה",
        "הראה רשימה",
        "מה יש ברשימה?",
        "show list",
        "מה יש לקנות?",
        "מה נשאר ברשימה",
        "תראה לי מה צריך",
        "מה הרשימה",
        "רשימת קניות",
        "מה עוד צריך לקנות?",
        "תראה רשימה",
        "מה קניתי?",
        "מה צריך?",
        "what's on the list?",
    ],
    "CLEAR_LIST": [
        "סיימתי קניות",
        "קניתי הכל",
        "clear list",
        "הכל נקנה",
        "סיימנו",
        "הרשימה הושלמה",
        "קניתי את הכל",
        "גמרתי את הקניות",
        "clear",
        "הכל קנוי",
        "נגמרו הקניות",
        "סגרנו רשימה",
    ],
    "REMOVE_ITEM": [
        "תמחק את החלב",
        "תוריד ביצים",
        "הורד עגבניות מהרשימה",
        "מחק לחם",
        "תסיר את הבננות",
        "הסר גבינה",
        "תוריד את התפוחים",
        "לא צריך יותר חמאה",
        "תמחק שוקולד",
        "תוריד את הקפה מהרשימה",
        "תמחק פסטה",
        "הסר שמיר מהרשימה",
    ],
    "PARTIAL_COMPLETION": [
        "קניתי הכל חוץ מחלב",
        "קניתי חוץ מביצים ולחם",
        "לקחתי הכל מלבד גבינה ויוגורט",
        "קניתי הכל חוץ מבשר",
        "הכל נלקח חוץ מנייר טואלט",
        "קניתי הכל פרט לתפוחים ובננות",
        "קניתי את הכל חוץ מפסטה, רוטב וגבינה",
        "לקחתי את כל הרשימה חוץ מקפה",
        "הכל קנוי מלבד סבון",
        "קניתי כמעט הכל, חוץ מחלב וביצים",
    ],
    "GROUP_INFO": [
        "מי בקבוצה?",
        "פרטי קבוצה",
        "מי חבר בקבוצה",
        "הצג חברי קבוצה",
        "מי נמצא בקבוצה?",
        "תראה לי מי בקבוצה",
        "כמה אנשים בקבוצה?",
        "מה שם הקבוצה?",
        "חברי הקבוצה",
        "מי שייך לקבוצה",
    ],
    "GET_INVITE_CODE": [
        "תן לי קוד",
        "מה הקוד?",
        "תן לי את קוד ההזמנה",
        "מה הקוד שלי?",
        "הצג קוד הזמנה",
        "תראה לי את הקוד",
        "יש לי קוד?",
        "מה קוד ההזמנה?",
        "אני רוצה לראות את הקוד",
        "תן קוד",
    ],
    "CREATE_INVITE": [
        "צור קוד הזמנה",
        "תייצר קוד הזמנה",
        "אני רוצה להזמין מישהו",
        "צור קוד חדש",
        "תייצר לי קוד להזמנת חבר",
        "אני רוצה להוסיף בן משפחה",
        "תייצר קוד",
        "צור הזמנה",
        "אני רוצה להזמין את אמא",
        "תייצר קוד הזמנה חדש",
    ],
    "RENAME_GROUP": [
        "שנה שם קבוצה למשפחת כהן",
        "תשנה את שם הקבוצה לבית לוי",
        "שנה שם לקניות שלנו",
        "רוצה לשנות שם קבוצה",
        "שם חדש לקבוצה: הקניות של יום שישי",
        "תקרא לקבוצה בית ישראלי",
        "שנה שם קבוצה",
        "עדכן שם קבוצה למשפחת דוד",
        "רוצה לשנות את שם הקבוצה",
        "שנה את השם לקניות משותפות",
    ],
    "LEAVE_GROUP": [
        "עזוב קבוצה",
        "צא מהקבוצה",
        "אני רוצה לעזוב",
        "תוציא אותי מהקבוצה",
        "אני עוזב את הקבוצה",
        "רוצה לצאת מהקבוצה",
        "תעזוב את הקבוצה",
        "אני לא רוצה להיות בקבוצה",
        "הסר אותי מהקבוצה",
        "צאי מהקבוצה",
    ],
    "NOTIFICATION_SETTINGS": [
        "בטל התראות",
        "כבה התראות",
        "השתק התראות",
        "הפעל התראות",
        "הדלק התראות",
        "הגדרות התראות",
        "סטטוס התראות",
        "מצב התראות",
        "אני לא רוצה התראות",
        "תפסיק לשלוח לי עדכונים",
        "רוצה לקבל התראות",
        "תדליק חזרה את ההתראות",
    ],
    "REVOKE_INVITE": [
        "בטל קוד הזמנה",
        "מחק קוד",
        "תבטל את קוד ההזמנה",
        "בטל את הקוד",
        "תמחק קוד הזמנה",
        "רוצה לבטל קוד",
        "הסר קוד הזמנה",
        "תבטל קוד",
        "בטל הזמנה",
        "מחק את הקוד",
    ],
    "RECIPE_SEARCH": [
        "תכין לי רשימה לשקשוקה",
        "מה צריך לחומוס?",
        "תן לי מתכון לפסטה",
        "רשימה לעוגת שוקולד",
        "מה צריך בשביל לזניה?",
        "תכין רשימת קניות לסלט יווני",
        "מה קונים בשביל שניצל?",
        "מה צריך לפנקייקים",
        "רשימה למרק עדשים",
        "תכין רשימה לפיצה",
    ],
    "UPDATE_QUANTITY": [
        "בעצם 5 חלב, לא 3",
        "שנה חלב ל-2",
        "תעדכן ביצים ל-3 קרטונים",
        "התכוונתי 4 ביצים",
        "תשנה את הכמות של הלחם ל-2",
        "בעצם צריך 2 קילו עגבניות",
        "עדכן כמות חלב ל-4",
        "שנה 3 חלב ולא 1",
        "תעדכן: 2 לחם במקום 1",
        "בעצם צריך עוד חלב, סה״כ 3",
    ],
    "BUG_REPORT": [
        "/bug הבוט לא עובד",
        "/bug לא מקבל תשובה",
        "/bug הרשימה ריקה אבל הוספתי פריטים",
        "/bug שגיאה כשאני מוסיף פריט",
        "/bug",
        "/bug ההתראות לא עובדות",
        "/bug הלינק לא נפתח",
        "/bug הקבוצה לא מתעדכנת",
        "/bug לא יכול להצטרף לקבוצה",
        "/bug הבוט מגיב באנגלית",
    ],
    "OTHER": [
        "מה מזג האוויר?",
        "שלום",
        "תודה",
        "מי אתה?",
        "ספר לי בדיחה",
        "מה השעה?",
        "איך אתה?",
        "היי",
        "בוקר טוב",
        "לילה טוב",
        "מה נשמע?",
        "עזרה",
        "מה אתה יודע לעשות?",
        "אני לא מבין",
        "test",
    ],
}

_DIVERSIFY_PROMPT = (
    "You are helping create training data for a Hebrew grocery shopping bot.\n"
    "Given the example message below, generate {n} UNIQUE paraphrases in Hebrew.\n"
    "Rules:\n"
    "- Vary formality (casual, formal, slang)\n"
    "- Add occasional typos (1 in 3 messages)\n"
    "- Mix Hebrew and English occasionally\n"
    "- Use short/long forms\n"
    "- Add emoji occasionally\n"
    "- Keep the SAME intent\n"
    "Return ONLY the paraphrases, one per line, no numbering.\n\n"
    "Example: {example}"
)


async def _diversify_seeds(
    client: httpx.AsyncClient,
    api_key: str,
    intent: str,
    seeds: list[str],
    variations: int,
) -> list[dict[str, str | int]]:
    """Paraphrase a SET of seeds for one intent.

    Returns samples enriched with `source_seed` so the caller can assert
    train/test seed disjointness. The `source_seed` field is dropped before
    writing the JSONL output.
    """
    samples: list[dict[str, str | int]] = []
    label_id = LABEL2ID[intent]

    # Add seeds themselves first
    for seed in seeds:
        samples.append({"text": seed, "label": label_id, "source_seed": seed})

    for seed in seeds:
        prompt = _DIVERSIFY_PROMPT.format(n=variations, example=seed)
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.9,
                    "max_tokens": 1000,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            for line in text.strip().split("\n"):
                line = line.strip().lstrip("0123456789.-) ")
                if line:
                    samples.append(
                        {"text": line, "label": label_id, "source_seed": seed}
                    )
        except Exception:
            logger.exception("Failed to diversify seed for %s: %s", intent, seed)

    return samples


async def generate_all(
    api_key: str,
    output_dir: Path,
    variations: int,
    test_seeds_per_intent: int,
    seed: int,
) -> None:
    """Generate training data with a SEED-LEVEL train/test split.

    For each intent, the seed templates are shuffled and `test_seeds_per_intent`
    are held out before paraphrasing. The two sets are paraphrased separately so
    no paraphrase of a held-out seed can land in the train set. Asserts seed
    disjointness at the end.

    An earlier version of this script split AFTER paraphrasing, which leaked
    paraphrases of the same seed into both splits and inflated the reported
    accuracy. Splitting at the seed level fixes that.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Decide the train/test seed lists per intent up front
    intent_split: dict[str, tuple[list[str], list[str]]] = {}
    for intent, all_seeds in SEED_TEMPLATES.items():
        if test_seeds_per_intent >= len(all_seeds):
            raise SystemExit(
                f"{intent}: test_seeds_per_intent={test_seeds_per_intent} "
                f"≥ available seeds ({len(all_seeds)}); reduce or add seeds"
            )
        shuffled = list(all_seeds)
        rng.shuffle(shuffled)
        intent_split[intent] = (
            shuffled[test_seeds_per_intent:],  # train_seeds
            shuffled[:test_seeds_per_intent],  # test_seeds
        )

    train_samples: list[dict[str, str | int]] = []
    test_samples: list[dict[str, str | int]] = []

    async with httpx.AsyncClient() as client:
        train_tasks = [
            _diversify_seeds(client, api_key, intent, train_seeds, variations)
            for intent, (train_seeds, _) in intent_split.items()
        ]
        test_tasks = [
            _diversify_seeds(client, api_key, intent, test_seeds, variations)
            for intent, (_, test_seeds) in intent_split.items()
        ]
        train_results = await asyncio.gather(*train_tasks)
        test_results = await asyncio.gather(*test_tasks)

    for batch in train_results:
        train_samples.extend(batch)
    for batch in test_results:
        test_samples.extend(batch)

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    # Assert: zero seed overlap (the entire point of the seed-level split).
    train_seed_set = {s["source_seed"] for s in train_samples}
    test_seed_set = {s["source_seed"] for s in test_samples}
    overlap = train_seed_set & test_seed_set
    if overlap:
        raise SystemExit(
            f"FATAL: {len(overlap)} seeds appear in BOTH train and test:\n"
            f"  {sorted(overlap)[:5]}..."
        )
    logger.info(
        "Seed-level split: %d train seeds, %d test seeds, 0 overlap",
        len(train_seed_set),
        len(test_seed_set),
    )

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    seed_manifest_path = output_dir / "split_manifest.json"

    # Strip `source_seed` from JSONL output to keep the schema compatible with
    # train_classifier.py and the downstream HF Datasets upload.
    for path, data in [(train_path, train_samples), (test_path, test_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                row = {"text": sample["text"], "label": sample["label"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Persist the per-intent split for transparency / reproducibility.
    seed_manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "test_seeds_per_intent": test_seeds_per_intent,
                "by_intent": {
                    intent: {"train": train_s, "test": test_s}
                    for intent, (train_s, test_s) in intent_split.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved %d train + %d test samples to %s (manifest: %s)",
        len(train_samples),
        len(test_samples),
        output_dir,
        seed_manifest_path.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HF classifier training data")
    parser.add_argument(
        "--output-dir", default="data", help="Output directory"
    )
    parser.add_argument(
        "--variations", type=int, default=10, help="Variations per seed"
    )
    parser.add_argument(
        "--test-seeds-per-intent",
        type=int,
        default=2,
        help=(
            "Seeds held out from training per intent (default 2). "
            "Replaces the older --test-ratio flag, which split after "
            "paraphrasing and leaked paraphrases across the two sides."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test seed selection",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "test_openai_key":
        logger.error("Set OPENAI_API_KEY environment variable")
        raise SystemExit(1)

    asyncio.run(
        generate_all(
            api_key,
            Path(args.output_dir),
            args.variations,
            args.test_seeds_per_intent,
            args.seed,
        )
    )


if __name__ == "__main__":
    main()
