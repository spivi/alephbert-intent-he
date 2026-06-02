# Evaluation Report: AlephBERT Hebrew Intent Classifier

_Generated: 2026-05-25 03:59 UTC_

## Headline numbers (3-run variance, seeds 42, 43, 44)

- Accuracy: 0.7424 ± 0.0031
- Weighted F1: 0.7357 ± 0.0039
- Macro F1: 0.7357 ± 0.0039
- Test samples: 374 (22 per intent), all paraphrases of seeds held out from training

## Baselines comparison

Every row is measured on the same 374-message test set. The column that matters is
cost: a zero-shot LLM charges per message, the fine-tune does not.

| Approach | Accuracy | Weighted F1 | Macro F1 | Cost per 1,000 messages |
|----------|---------:|------------:|---------:|-------------------------|
| Random | 0.0668 | 0.0637 | 0.0637 | $0 |
| Majority class | 0.0588 | 0.0065 | 0.0065 | $0 |
| Keyword regex (hand-written) | 0.2487 | 0.2834 | 0.2834 | $0 |
| GPT-4o-mini zero-shot | 0.5722 | 0.5916 | 0.5916 | about $0.05 (gpt-4o-mini, Jan 2026 pricing, ~250-token prompt) |
| AlephBERT fine-tune (this model) | 0.7424 ± 0.0031 | 0.7357 ± 0.0039 | 0.7357 ± 0.0039 | $0 |

The fine-tune is about 17 points more accurate than GPT-4o-mini zero-shot, and once
trained it costs nothing per message. For a narrow task like this, a small Hebrew
model is enough to beat a general-purpose LLM.

## Per-intent F1 (3-run mean ± std)

| Intent | F1 (mean ± std) | Support |
|--------|----------------:|--------:|
| `GROCERY_REQUEST` | 0.767 ± 0.023 | 22 |
| `RECIPE_URL` | 0.740 ± 0.029 | 22 |
| `LIST_QUERY` | 0.793 ± 0.054 | 22 |
| `CLEAR_LIST` | 0.680 ± 0.026 | 22 |
| `REMOVE_ITEM` | 0.712 ± 0.061 | 22 |
| `PARTIAL_COMPLETION` | 0.825 ± 0.040 | 22 |
| `GROUP_INFO` | 0.590 ± 0.053 | 22 |
| `GET_INVITE_CODE` | 0.841 ± 0.029 | 22 |
| `CREATE_INVITE` | 0.635 ± 0.069 | 22 |
| `RENAME_GROUP` | 0.993 ± 0.013 | 22 |
| `LEAVE_GROUP` | 0.791 ± 0.031 | 22 |
| `NOTIFICATION_SETTINGS` | 0.615 ± 0.014 | 22 |
| `REVOKE_INVITE` | 0.900 ± 0.011 | 22 |
| `RECIPE_SEARCH` | 0.804 ± 0.028 | 22 |
| `UPDATE_QUANTITY` | 0.977 ± 0.001 | 22 |
| `BUG_REPORT` | 0.340 ± 0.011 | 22 |
| `OTHER` | 0.505 ± 0.009 | 22 |

## Confusion matrix

![Confusion matrix from the seed 42 run](confusion_matrix.png)

## Methodology

- Train/test split: seed-level. For every intent, 2 seeds are held out before
  paraphrasing, and the test set contains only paraphrases of those held-out seeds.
  No seed appears on both sides (the data script asserts this at runtime).
- Multi-run variance: 3 training runs with seeds 42, 43, and 44 on the same data,
  changing only the training RNG. Accuracy values: 0.7406, 0.7460, 0.7406.
- Inference: ONNX runtime on CPU, validated to match the PyTorch logits within
  `atol=1e-4`.
- Reproducible:
  ```bash
  python scripts/generate_training_data.py --seed 42 --test-seeds-per-intent 2
  python scripts/multi_run_train.py --n-runs 3 --base-seed 42
  ```
