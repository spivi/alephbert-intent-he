# AlephBERT Hebrew Intent Classifier

A small, end-to-end example of fine-tuning a Hebrew language model to read short
messages and decide what the user wants. It is a learning project: the point is
to show the whole process, from making training data to shipping a live demo, in
a way you can follow step by step and adapt to your own idea.

If you are new to this, that is fine. The whole thing runs on a laptop or on a
free Google Colab GPU, and every command below is meant to be copied and run.

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-spivi87%2Falephbert--intent--he-blue)](https://huggingface.co/spivi87/alephbert-intent-he)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)

- **Trained model:** [spivi87/alephbert-intent-he](https://huggingface.co/spivi87/alephbert-intent-he)
- **Live demo (try it in the browser):** [spivi87/alephbert-intent-he-demo](https://huggingface.co/spaces/spivi87/alephbert-intent-he-demo)
- **Sample of the training data:** [`data/sample.jsonl`](./data/sample.jsonl)

## What you will learn

The task is to take a short Hebrew message, like "תוסיף חלב וביצים" ("add milk and
eggs"), and label it with one of 17 intents (add an item, show the list, clear the
list, and so on). Along the way the repo shows you how to:

1. Generate your own training data with an LLM, instead of collecting and labelling
   real messages by hand.
2. Split the data so you do not fool yourself with an accuracy number that is too
   good to be true.
3. Fine-tune a Hebrew BERT model with the HuggingFace `Trainer`.
4. Measure the result honestly and compare it against simpler options.
5. Export the model and put up a small demo.

## The result that matters

The interesting comparison is not "is this model fast". It is "do I need to pay a
cloud LLM for every message, or can I train a small model once and then run it for
free". Both numbers below are on the same 374-row test set (see
[EVALUATION.md](./EVALUATION.md)):

| Approach | Accuracy | Cost per 1,000 messages |
|----------|---------:|-------------------------|
| Random guessing | 6.7% | $0 |
| Always guess the most common label | 5.9% | $0 |
| Keyword rules (16 hand-written rules) | 24.9% | $0 |
| GPT-4o-mini, zero-shot prompt | 57.2% | about $0.05 |
| AlephBERT fine-tune (this repo) | 76.2% | $0 |

So the small fine-tuned model is about 19 points more accurate than asking
GPT-4o-mini directly, and after you train it once it costs nothing to run. This is
a single training run measured on the held-out test set.

## Quickstart: use the trained model

```python
from transformers import pipeline

clf = pipeline("text-classification", model="spivi87/alephbert-intent-he", top_k=3)
clf("תוסיף חלב וביצים")
# [{'label': 'GROCERY_REQUEST', 'score': 0.98}, ...]
```

The model gives back readable label names (not `LABEL_0`). The mapping from
numbers to names is saved inside `config.json`, so you get `GROCERY_REQUEST`
straight away.

## Reproduce it from scratch

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # only needed to generate the synthetic data

# 1. Generate synthetic training data (the split is explained in "How it works")
python scripts/generate_training_data.py --output-dir data --test-seeds-per-intent 2 --seed 42

# 2. Fine-tune AlephBERT
python scripts/train_classifier.py --data-dir data --output-dir model --seed 42

# 3. Export the model to ONNX (set dynamo=False; see the note in export_onnx.py)
python scripts/export_onnx.py --model-dir model --output model.onnx

# 4. Bake the label names into config.json so pipeline() returns readable labels
python scripts/bake_id2label.py --input model --output model-publish

# 5. Evaluate on the held-out test split
python scripts/evaluate.py --onnx-model model.onnx --tokenizer-dir model \
    --test-data data/test.jsonl --output eval.json
python scripts/generate_evaluation_report.py --input eval.json \
    --md-out EVALUATION.md --png-out confusion_matrix.png
```

To see how much the accuracy moves between training runs:

```bash
python scripts/multi_run_train.py --data-dir data --output-dir runs --n-runs 3 --base-seed 42
```

To compare against the simpler baselines:

```bash
python scripts/baselines/random_baseline.py   --test-data data/test.jsonl --output b_random.json
python scripts/baselines/majority_baseline.py --train-data data/train.jsonl --test-data data/test.jsonl --output b_majority.json
python scripts/baselines/regex_baseline.py    --test-data data/test.jsonl --output b_regex.json
python scripts/baselines/gpt_zeroshot_baseline.py --test-data data/test.jsonl --output b_gpt.json  # needs OPENAI_API_KEY
```

## How it works

1. **Synthetic data, no manual labelling.** `generate_training_data.py` starts
   from a few hand-written Hebrew example sentences per intent (the "seeds") and
   asks GPT-4o-mini to rewrite each one into about 10 variations, changing the
   wording, adding typos, emoji, and a bit of English. There are no real user
   messages in here, so there is no private data to worry about. A small sample
   of the result is in [`data/sample.jsonl`](./data/sample.jsonl) so you can see
   the format. After the first version I noticed real gaps (for example "buy X"
   requests like "תקנה חלב" were misread), so I added a batch of hand-authored
   examples to cover the missing phrasings and retrained. Testing the model and
   feeding the failures back into the data is most of the work.
2. **A split that does not cheat.** The train/test split happens at the seed
   level, before the rewriting step. A couple of seeds per intent are kept aside
   completely, and the test set only contains rewrites of those held-out seeds.
   This matters: if rewrites of the same seed end up in both train and test, the
   model has basically already seen the test questions, and the accuracy looks
   better than it really is. The script checks at the end that no seed appears in
   both sides.
3. **Fine-tune.** `train_classifier.py` fine-tunes `onlplab/alephbert-base` with
   the HuggingFace `Trainer` (AdamW, learning rate 2e-5, batch size 16, up to 128
   tokens per message, early stopping). All the random seeds are fixed with
   `--seed` so you can get the same result again.
4. **ONNX export.** `export_onnx.py` exports the trained model to ONNX, a format
   you can run without PyTorch installed. It uses the older TorchScript exporter
   on purpose (`dynamo=False`); the newer exporter quietly dropped weights for
   BERT-style models when this was written. The export is checked against the
   PyTorch output to make sure they agree.

## Adapt it to your own task

The shopping intents are just the example. The same recipe works for any Hebrew
text-classification task. To point it at your own problem:

1. Edit `INTENT_LABELS` and `EN_DESCRIPTIONS` in `scripts/label_map.py`.
2. Replace `SEED_TEMPLATES` in `scripts/generate_training_data.py` with a few
   example sentences per class for your own domain.
3. Re-run the steps above. Adjust `--test-seeds-per-intent`, `--variations`, and
   the confidence threshold to match how many classes and how much data you have.

### Fine-tune on your own data in Colab (no setup)

If you already have a labelled CSV (with `text` and `label` columns), you can skip
the synthetic-data step and fine-tune straight away in your browser on a free T4
GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spivi/alephbert-intent-he/blob/main/finetune.ipynb)

The notebook ([`finetune.ipynb`](./finetune.ipynb)) reads your CSV, builds the
label map for you, fine-tunes AlephBERT, prints a per-class report, and exports to
ONNX. End to end it takes about 10 to 25 minutes.

## Good to know before you rely on it

- It was trained on shopping and grocery messages, so treat it as a starting point
  for your own fine-tune, not as a general-purpose Hebrew classifier.
- The catch-all `OTHER` class is the weakest one (F1 around 0.5). In a real product
  it helps to send any low-confidence prediction (`score < 0.7`) to an LLM as a
  fallback.
- The training data is synthetic and fairly clean, so very noisy or unusual
  phrasing may be classified less reliably than the examples in the test set.
- It expects Hebrew, in short messages of up to 128 tokens.

## Credits

This project stands on top of **AlephBERT**, a Hebrew BERT model from the ONLP Lab
at Bar-Ilan University. AlephBERT was built by Amit Seker, Elron Bandel, Dan
Bareket, Idan Brusilovsky, Refael Greenfeld, and Reut Tsarfaty. All the heavy
lifting of learning the Hebrew language was done by their model. This repo only
fine-tunes it on a small intent-classification task, and it would not exist
without their work. Thank you to the ONLP Lab team.

If AlephBERT is useful to you, please cite their paper:

```bibtex
@inproceedings{seker-etal-2022-alephbert,
    title     = "{A}leph{BERT}: Language Model Pre-training and Evaluation from Sub-Word to Sentence Level",
    author    = "Seker, Amit and Bandel, Elron and Bareket, Dan and Brusilovsky, Idan and Greenfeld, Refael and Tsarfaty, Reut",
    booktitle = "Proceedings of the 60th Annual Meeting of the ACL",
    year      = "2022",
    url       = "https://aclanthology.org/2022.acl-long.4",
}
```

## License

Apache 2.0 (see [LICENSE](./LICENSE)). Built on
[`onlplab/alephbert-base`](https://huggingface.co/onlplab/alephbert-base), which
is also Apache 2.0.
