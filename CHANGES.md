# CHANGES

Change log for the improvement pass. Backup of the pre-change project lives at
`../KGCBC` (copied before any edits).

## Critical finding: the "space" leak

Before any model work, an analysis of `train_data.csv` showed:

| Class | rows with a space | space_frac mean |
|---|---|---|
| Wrong (0) | 10,053 / 10,053 (100%) | 0.1533 |
| Right (1) | 5,019 / 10,053 (49.9%) | 0.0098 |

The Wrong sentences are scrambled tokens that were written **space-separated**,
while most Right sentences are fully joined (normal Khmer orthography). A model
could reach ~75% accuracy on the space feature alone, and `oov_ratio` was
inflated on Wrong rows as a side effect. The reported 89% test accuracy was
therefore measuring spelling/space detection more than grammar.

Fix: `scripts/generate_errors.py` regenerates negatives joined **without
spaces** (`"".join(tokens)`), so spacing is never a predictive feature.

## Data

- `scripts/generate_errors.py` — new. Builds a balanced labeled dataset from a
  raw Khmer corpus with an 8-type error taxonomy (shuffle, swap_adjacent,
  drop_token, drop_subject, front_verb, drop_particle, duplicate, fragment).
  Propagates `doc_id` so train/test can be split by source document (prevents
  a sentence leaking together with its corrupted twin). Accepts a plain text
  corpus (tagged via `khmer-pos-roberta`) or an already-tagged CSV.

## Model

- `src/models/gru.py` — `ImprovedGRUClassifier` now optionally takes per-token
  POS embeddings (`pos_vocab_size`, `pos_embedding_dim`), concatenated with the
  FastText vectors before the GRU. POS tags were previously collapsed into one
  scalar; now the tag sequence itself is a grammar signal.
- `src/config.py` — added `POS_TAGS` (24-tag tagset from `train_data.csv`),
  `POS_EMBEDDING_DIM=32`, `CLASS_WEIGHTS=[1.3, 1.0]`.
- `train_transformer.py` — new. Fine-tunes `seanghay/xlm-roberta-khmer-small`
  (Khmer-trained RoBERTa, 49.7M params) directly on raw text, replacing the
  FastText+GRU+manual-features stack. Saves a checkpoint with a tuned decision
  threshold. Highest expected accuracy ceiling for this low-resource task.

## Training / inference

- `train.py` — fixed `NameError`: `load_and_split_data` returns were unpacked
  as `_, _, _` while `X_train_scaled`/`X_val_scaled` were used later. The scaled
  tensors are now bound, so feature fusion actually trains.
- `train.py` — weighted `CrossEntropyLoss` (`CLASS_WEIGHTS`) to counter the
  over-predict-"Right" false-positive bias; a validation threshold sweep
  maximizes F1 and the best threshold is saved in the checkpoint.
- `src/models/train_utils.py` — added `predict_probs_with_features` and
  `find_best_threshold`; all train/eval loops now thread `pos_ids` to the model.
- `src/models/predictor.py` — applies the checkpoint threshold at inference
  (`P(Right) >= threshold`) and feeds POS ids to the model.
- `src/data/dataset.py` — dataset now returns `(embeddings, pos_ids, label,
  length)`; `collate_batch` pads POS ids.
- `src/pipeline/feature_pipeline.py` — `create_temp_dataset` accepts POS tags.

## Consistency fixes

- `main.py` (API) was loading the old plain `GRUClassifier` while the saved
  checkpoint and every other script used `ImprovedGRUClassifier`. Now uses the
  improved architecture + feature fusion, and reads the saved threshold.
- The checked-in `gru_model.pth` was the stale plain-GRU checkpoint
  (state-dict keys `gru.*`/`fc.*`, no `attention.*`/`classifier.*`). Retrain to
  regenerate it for the improved architecture.

## Evaluation

- `evaluate.py` — removed duplicate `import torch`, dropped unused imports,
  switched to threshold-aware test evaluation, and fixed inverted label names
  (`target_names=["Wrong","Right"]`; 0=Wrong, 1=Right — previously labeled
  index 0 as "Right").
