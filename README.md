# AFP_EXP — Automatic Fact Checker (FEVER)

An NLP experiment that builds an **automatic fact-checking pipeline** on top of the
[FEVER](https://fever.ai/) (Fact Extraction and VERification) dataset. Given a natural-language
**claim** and its supporting **evidence sentence(s)** extracted from Wikipedia, the project
fine-tunes a **BERT** model to classify the claim as one of:

| Label | Meaning |
|-------|---------|
| `SUPPORTS` | The evidence supports the claim |
| `REFUTES` | The evidence contradicts the claim |
| `NOT ENOUGH INFO` | The evidence is insufficient to verify the claim |

This is essentially a **Natural Language Inference (NLI)** task: the model reads the
`(claim, evidence)` pair and predicts the relationship between them.

> The core work lives in the notebook [`NLP_Fact_Checker.ipynb`](NLP_Fact_Checker.ipynb),
> originally developed in Google Colab.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [The FEVER Dataset](#the-fever-dataset)
- [Pipeline / How It Works](#pipeline--how-it-works)
- [Model](#model)
- [Getting Started](#getting-started)
- [Notes & Design Decisions](#notes--design-decisions)
- [Learning Notes](#learning-notes)
- [Possible Improvements](#possible-improvements)

---

## Project Overview

Fact-checking a claim automatically requires two broad capabilities:

1. **Evidence retrieval** — finding the relevant Wikipedia sentence(s) for a claim.
2. **Claim verification** — deciding whether that evidence supports, refutes, or is
   insufficient for the claim.

This project focuses on preparing clean `(claim, evidence, label)` training data from the raw
FEVER dumps and then fine-tuning a transformer classifier for the **verification** step.
Because FEVER already provides the ground-truth evidence pointers (Wikipedia page + sentence id)
for each claim, the pipeline resolves those pointers into actual sentence text and trains on
them directly.

---

## Repository Structure

```
AFP_EXP/
├── NLP_Fact_Checker.ipynb          # Main notebook: data prep + BERT fine-tuning
├── FEVER_dataset/
│   ├── train.jsonl                 # ~145k training claims
│   ├── paper_dev.jsonl             # ~10k dev claims
│   └── paper_test.jsonl            # ~10k test claims
├── DeepLearningMethodTechniques.txt # Notes: RNN → LSTM → Attention → Transformers
├── Learnings and Observation.txt    # Notes: NLP preprocessing & project roadmap
├── Txt_PreProcessing_Guideline.png  # Reference diagram for text preprocessing
└── README.md
```

---

## The FEVER Dataset

Each line in the `FEVER_dataset/*.jsonl` files is a JSON record describing one claim:

```json
{
  "id": 75397,
  "verifiable": "VERIFIABLE",
  "label": "SUPPORTS",
  "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
  "evidence": [[[92206, 104971, "Nikolaj_Coster-Waldau", 7],
               [92206, 104971, "Fox_Broadcasting_Company", 0]]]
}
```

Field meaning:

- **`claim`** — the statement to be fact-checked.
- **`label`** — the gold verdict (`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`).
- **`evidence`** — nested lists of evidence pointers. Each innermost item is
  `[annotation_id, evidence_id, wiki_page_title, sentence_id]`, pointing to a specific
  **sentence within a specific Wikipedia page** that serves as evidence. For
  `NOT ENOUGH INFO` claims, the page/sentence pointers are `null`.

In addition to the claims, the notebook pulls the **`wiki_pages`** configuration of FEVER
(via Hugging Face `datasets`), which contains the full text of Wikipedia articles, split into
individually-numbered sentences. These are needed to turn a `(page, sentence_id)` pointer into
actual evidence **text**.

---

## Pipeline / How It Works

The notebook implements the following end-to-end flow:

1. **Load data**
   - FEVER **claims** (`train` and `labelled_dev` splits) via `load_dataset("fever", "v1.0")`.
   - FEVER **`wiki_pages`** (streamed) for the Wikipedia article text.

2. **Filter Wikipedia pages**
   - Collect the set of distinct `evidence_wiki_url` page titles referenced by the claims, then
     stream through the (large) `wiki_pages` corpus and keep only the pages that are actually
     referenced. The filtered set is cached to Google Drive as `filtered_pages.pkl` to avoid
     re-streaming.

3. **Index Wikipedia pages** (`indexing_wiki_pages`)
   - Build a nested dictionary `wiki_index[page_id][sentence_id] = sentence_text` by splitting
     each page's `lines` field on newlines and tabs.

4. **Map claims → evidence sentences** (`mapping_wiki_pages_sentence`)
   - For every claim, look up its `(evidence_wiki_url, evidence_sentence_id)` in the index and
     attach the resolved sentence as a new `corr_sentence` column. Done for both train and
     validation splits.

5. **Clean & aggregate the data** (pandas)
   - Drop degenerate records (e.g. `SUPPORTS`/`REFUTES` claims that end up with no usable
     evidence sentence, or duplicate/contradictory annotation rows).
   - **Aggregate** multiple evidence sentences belonging to the same claim `id` into a single
     joined `corr_sentence` string, producing one `(id, label, claim, corr_sentence)` row per
     claim.

6. **Tokenization** (`handling_tokenisation` / `tokenize`)
   - Uses the `bert-base-uncased` tokenizer on the `(claim, corr_sentence)` sentence pair.
   - **Short inputs** (≤ 512 tokens): padded/truncated to `max_length=512`.
   - **Long inputs** (> 512 tokens): split into overlapping **chunks** using
     `stride=128` and `return_overflowing_tokens=True`, so no evidence is silently dropped.
     Each chunk inherits the claim's label.
   - Labels are encoded via `LABEL_CODING = {'SUPPORTS':0, 'REFUTES':1, 'NOT ENOUGH INFO':2}`.

7. **Model fine-tuning** (Hugging Face `Trainer`)
   - `BertForSequenceClassification` (`bert-base-uncased`, `num_labels=3`) is fine-tuned on the
     tokenized pairs and evaluated on the validation split each epoch.

---

## Model

- **Base model:** `bert-base-uncased`
- **Head:** sequence-classification head with 3 output labels
- **Task framing:** sentence-pair classification (`claim` = sentence A, `evidence` = sentence B)

Key training arguments used in the notebook:

| Argument | Value |
|----------|-------|
| `learning_rate` | `2e-5` |
| `per_device_train_batch_size` | `8` |
| `per_device_eval_batch_size` | `16` |
| `num_train_epochs` | `3` |
| `weight_decay` | `0.01` |
| `warmup_steps` | `500` |
| `max_length` | `512` (stride `128` for long inputs) |
| `eval_strategy` / `save_strategy` | `epoch` |
| `load_best_model_at_end` | `True` (by `eval_accuracy`) |
| `fp16` | `True` (mixed precision on GPU) |

---

## Getting Started

The notebook was written for **Google Colab** (it mounts Google Drive and uses a GPU runtime),
but it can be adapted to run locally.

### Run in Colab (recommended)
1. Open [`NLP_Fact_Checker.ipynb`](NLP_Fact_Checker.ipynb) in Google Colab.
2. Select a **GPU** runtime (`Runtime → Change runtime type → GPU`).
3. Run the cells top to bottom. The first run of the Wikipedia-page filtering step is slow;
   the notebook caches the result to Drive (`filtered_pages.pkl`) for subsequent runs.

### Run locally
1. Create an environment and install dependencies:
   ```bash
   pip install datasets transformers torch pandas tqdm wikipedia
   ```
2. Remove/replace the Colab-specific `google.colab.drive` mounting code and point the pickle
   cache at a local path.
3. Ensure you have a CUDA-capable GPU for fine-tuning (otherwise disable `fp16=True` and expect
   long CPU training times).

---

## Notes & Design Decisions

- **Why cache `filtered_pages.pkl`?** The FEVER `wiki_pages` corpus is very large; streaming and
  filtering it every run is expensive, so the referenced subset is pickled once and reloaded.
- **Handling long evidence.** Rather than truncating long `(claim, evidence)` pairs, they are
  chunked with a 128-token stride so the model still sees all the evidence across chunks.
- **Data cleaning matters.** A significant portion of the notebook is dedicated to removing
  claims whose evidence sentences can't be resolved, deduplicating annotation rows, and merging
  multi-sentence evidence — clean pairs are essential for a reliable classifier.
- **Unicode / non-ASCII page titles.** Some Wikipedia titles contain accented characters
  (e.g. `Mélanie_Laurent`). The notebook includes commented-out normalization helpers
  (`unicodedata.normalize('NFC', ...)`) for reconciling these mismatches.

---

## Learning Notes

The two accompanying text files capture the author's study notes behind the design:

- **[`DeepLearningMethodTechniques.txt`](DeepLearningMethodTechniques.txt)** — the progression
  from RNNs → LSTMs → the attention mechanism → Transformers, and why transformer-based models
  (with parallelized, full-sentence input) are used for evidence retrieval/verification.
- **[`Learnings and Observation.txt`](Learnings%20and%20Observation.txt)** — NLP preprocessing
  steps (lowercasing, stemming, lemmatization, stop-word removal, normalization, noise removal,
  augmentation) and the project roadmap (load → index wiki pages → map claims to sentences →
  sentence embeddings).

---

## Possible Improvements

- Add an **evidence-retrieval** stage (e.g. sentence embeddings / dense retrieval) so the system
  can fact-check claims without gold evidence pointers — the realistic end-to-end setting.
- Report **evaluation metrics** (accuracy, per-class precision/recall/F1, confusion matrix) and
  save them.
- Handle **class imbalance** across `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`.
- Experiment with stronger NLI backbones (RoBERTa, DeBERTa) and proper handling of the
  multi-chunk predictions at inference time (aggregating chunk logits per claim).
- Refactor the notebook into reusable Python modules with a `requirements.txt`.
```