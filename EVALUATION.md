# Evaluation Report: AlephBERT Hebrew Intent Classifier

## Summary (mean over 3 seeds: 42, 43, 44)

- Accuracy: 0.773 ± 0.008
- Macro F1: 0.763 ± 0.008
- Test samples: 374 (22 per intent), held out at the seed level from training
- The published checkpoint is the seed 42 run.

The train and test data are synthetic paraphrases, not real chat traffic. The
split holds out source seeds before paraphrasing to avoid leakage, but the numbers
are still a controlled experiment, not proof of real-world accuracy.

## Baselines comparison

Every row is measured on the same 374-message held-out test set.

| Approach | Accuracy | Cost per 1,000 messages |
|----------|---------:|-------------------------|
| Random | 0.0668 | $0 |
| Majority class | 0.0588 | $0 |
| Keyword regex (hand-written) | 0.2487 | $0 |
| GPT-4o-mini zero-shot | 0.5722 | about $0.05 (estimate, see note) |
| AlephBERT fine-tune (this model, seed 42) | 0.7834 | $0 |

On this narrow synthetic task the fine-tune scored about 20 points higher than my
zero-shot GPT-4o-mini baseline on the same split. That is not a general claim that
it beats GPT-4o-mini; it means a small task-specific model can be cheap and private
for a narrow, repeated task. The GPT-4o-mini cost is an estimate for my zero-shot
prompt and these short messages; OpenAI bills per token, not per message, so your
cost will vary with prompt length, label count, and output format.

## Robustness check: terse single-item requests

An early version misread short single-item requests (for example "תקנה ביצים",
buy eggs) as REMOVE_ITEM, because single-item phrases were over-represented in the
remove examples. I added single-item add/buy/need examples across many item nouns
and retrained. A small hand-built check of 30 such cases (single-item buy / add /
need vs remove) now scores 100%.

## Per-intent metrics (seed 42)

| Intent | Precision | Recall | F1 | Support |
|--------|----------:|-------:|---:|--------:|
| `GROCERY_REQUEST` | 0.800 | 0.909 | 0.851 | 22 |
| `RECIPE_URL` | 0.900 | 0.818 | 0.857 | 22 |
| `LIST_QUERY` | 0.760 | 0.864 | 0.809 | 22 |
| `CLEAR_LIST` | 0.722 | 0.591 | 0.650 | 22 |
| `REMOVE_ITEM` | 0.750 | 0.818 | 0.783 | 22 |
| `PARTIAL_COMPLETION` | 0.909 | 0.909 | 0.909 | 22 |
| `GROUP_INFO` | 1.000 | 0.545 | 0.706 | 22 |
| `GET_INVITE_CODE` | 0.786 | 1.000 | 0.880 | 22 |
| `CREATE_INVITE` | 0.619 | 0.591 | 0.605 | 22 |
| `RENAME_GROUP` | 0.957 | 1.000 | 0.978 | 22 |
| `LEAVE_GROUP` | 0.714 | 0.909 | 0.800 | 22 |
| `NOTIFICATION_SETTINGS` | 0.857 | 0.545 | 0.667 | 22 |
| `REVOKE_INVITE` | 0.808 | 0.955 | 0.875 | 22 |
| `RECIPE_SEARCH` | 0.808 | 0.955 | 0.875 | 22 |
| `UPDATE_QUANTITY` | 1.000 | 0.955 | 0.977 | 22 |
| `BUG_REPORT` | 0.375 | 0.273 | 0.316 | 22 |
| `OTHER` | 0.600 | 0.682 | 0.638 | 22 |

## Confusion matrix

![Confusion matrix](confusion_matrix.png)

## Methodology

- Train/test split: seed-level. For every intent, 2 seeds are held out before
  paraphrasing, and the test set contains only paraphrases of those held-out seeds.
- Training data: LLM-generated paraphrases of Hebrew seed templates, plus
  hand-authored examples added to cover phrasings the first version missed.
- 3 training runs (seeds 42, 43, 44); the numbers above are the mean and standard
  deviation, and seed 42 is the published checkpoint.
