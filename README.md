# Khmer Grammar Checking (Binary Classification)

ITM 454 - Natural Language Processing course project. A Khmer Grammar Checking System formulated as a binary sentence-level classification problem: given a Khmer sentence, the model predicts whether it is grammatically **acceptable (Right)** or **unacceptable (Wrong)**.

> **Latest improvements** are documented in [CHANGES.md](CHANGES.md). Key points: per-token POS embeddings in the BiGRU, weighted loss + tuned decision threshold, a Khmer-transformer path (`train_transformer.py`), a data-generation script with an error taxonomy (`scripts/generate_errors.py`), and the fix for a dataset "space" leak that let the old model cheat (100% of Wrong rows contained a space vs 49.9% of Right rows).

The system follows a standard supervised NLP pipeline adapted for low-resource language settings. Text is tokenized and POS-tagged with [khmer-pos-roberta](https://huggingface.co/seanghay/khmer-pos-roberta). Features are manually engineered from the POS tags and FastText embeddings, then fed into a **BiGRU classifier** with attention pooling and feature fusion.

## Pipeline Architecture

```
Raw Khmer Sentence
        │
        ▼
┌───────────────────┐
│  POS Tagging      │  seanghay/khmer-pos-roberta
│  (tokens + POS)   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Feature Extraction│
│                   │
│  ┌─────────────┐  │
│  │ OOV Ratio   │  │  FastText vocabulary (spelling errors)
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ Grammar     │  │  Strict POS-sequence grammar score
│  │ Score       │  │  (penalizes no-verb, noun-only)
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ Sentence    │  │  Token count (short=fragment)
│  │ Length      │  │
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ POS         │  │  Unique POS / total tokens
│  │ Diversity   │  │  (sentence completeness proxy)
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ Avg Word    │  │  Mean char length per token
│  │ Length      │  │  (word complexity signal)
│  └─────────────┘  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  FastText         │  300-dim cc.km.300.bin
│  Embeddings       │  (per token, stacked)
└───────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│  BiGRU (2-layer, bidirectional)   │
│  Hidden dim: 256                   │
│  Output: sequence of hidden states │
└────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Attention Pooling + Mean    │  2 x (2*256) = 1024-dim
│  Pooling (concatenated)      │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Feature Fusion (concat)     │  append 5 scalar features
│  1024 + 5 = 1029-dim         │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  3-layer MLP Classifier      │
│  Linear → LayerNorm → ReLU   │
│  → Dropout → Linear → ...   │
│  → Linear(128, 2)            │
└──────────────────────────────┘
        │
        ▼
   [Right] / [Wrong]
```

## Project Structure

```
src/
├── __init__.py
├── config.py              # Hyperparameters, paths, feature column list
├── utils.py               # Data loading, splitting, scaling, seeding
├── data/
│   ├── __init__.py
│   ├── dataset.py          # KhmerTextDataset with embedding caching
│   └── pos_tagger.py       # KhmerPOSTagger (seanghay/khmer-pos-roberta)
├── features/
│   ├── __init__.py
│   ├── oov.py              # EmbeddingOOVCalculator — oov_ratio
│   ├── grammar.py          # SimplePOSGrammarExtractor — dep_grammar_score
│   └── structural.py       # StructuralFeatureExtractor — length, diversity, word length
├── models/
│   ├── __init__.py
│   ├── gru.py              # GRUClassifier, ImprovedGRUClassifier, AttentionPooling
│   ├── train_utils.py      # Training/evaluation loops, EarlyStopping
│   └── predictor.py        # SentencePredictor for inference
└── pipeline/
    ├── __init__.py
    └── feature_pipeline.py # FeaturePipeline orchestrator
train.py                    # Full training script
train_transformer.py        # Khmer-transformer fine-tuning (alternative path)
evaluate.py                 # Evaluation on test set
predict.py                  # Single-sentence prediction
main.py                     # FastAPI web server
scripts/
└── generate_errors.py      # Build a labeled dataset with an error taxonomy
requirements.txt
train_data.csv              # Pre-extracted feature CSV (20,106 sentences, balanced)
CHANGES.md                  # Change log for the improvement pass
README.md
```

## Features

| Feature | Source | Description |
|---|---|---|
| `oov_ratio` | FastText vocab | Fraction of tokens not in FastText vocabulary (spelling error indicator) |
| `dep_grammar_score` | POS tags | Strict grammar score (0.0–1.0) penalizing no-verb and noun-only sentences |
| `sentence_length` | Tokens | Number of tokens (short sentences tend to be fragments) |
| `pos_diversity` | POS tags | Unique POS tags / total tokens (sentence completeness proxy) |
| `avg_word_length` | Tokens | Mean character length per token (word complexity) |

### Removed Features (caused false-positive bias)

| Feature | Reason for Removal |
|---|---|
| `semantic_coherence` | High cosine similarity for ALL sentences (words are coherent even in bad grammar); not discriminative, computationally expensive |
| `grammar_oov_interaction` | Pure redundancy: `grammar * (1 - oov)`, no independent signal |
| `has_complete_clause` | Too permissive, almost every sentence with NN+VB scored 1 |

## Model

**BiGRU with Attention Pooling + Feature Fusion**
- 300-dim FastText embeddings (cc.km.300.bin) + 32-dim per-token POS embeddings
- 2-layer bidirectional GRU (hidden dim 256)
- Attention + mean pooling concatenated (1024-dim)
- Feature fusion: 5 scalar features concatenated to pooled representation
- 3-layer MLP classifier with LayerNorm and dropout (0.5)
- Weighted cross-entropy (`CLASS_WEIGHTS=[1.3, 1.0]`), early stopping (patience 3),
  gradient clipping (1.0), Adam optimizer (lr=1e-3, weight decay=1e-5)
- Tuned decision threshold (max F1 on validation), saved in the checkpoint and
  applied at inference

**Alternative: Khmer Transformer** (`train_transformer.py`)
- Fine-tunes `seanghay/xlm-roberta-khmer-small` (Khmer-trained RoBERTa, 49.7M
  params) directly on raw text — no FastText, no manual features.
- Highest expected accuracy ceiling for this low-resource task.

### Results (Test Set — old feature set)

```
Accuracy:  88.79%
Precision: 85.01%
Recall:    94.18%
F1-Score:  89.36%
```

> **Note:** The model exhibited a false-positive bias (94% recall, 85% precision) due to non-discriminative features. Retraining with the improved feature set above should reduce false positives and improve precision.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py

# Fine-tune the Khmer transformer instead (no feature pipeline needed)
python train_transformer.py

# Generate a new labeled dataset from a raw Khmer corpus (error taxonomy)
python scripts/generate_errors.py corpus.txt -o data_augmented.csv

# Evaluate
python evaluate.py

# Predict a single sentence
python predict.py "អ៊ីតាលីបានឈ្នះលើព័រទុយហ្គាល់ 31-5"

# Run API server
uvicorn main:app --reload
```

## Data

The training data (`train_data.csv`) contains 20,106 sentences (50/50 balanced) extracted from the ALT Parallel Corpus. Each row includes tokens, POS tags, engineered features, and a binary `sentence_correct` label (1 = correct, 0 = incorrect).

## Requirements

- PyTorch >= 2.0
- transformers (for khmer-pos-roberta)
- fasttext (for cc.km.300.bin embeddings)
- scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm
- fastapi, uvicorn (for API server)
