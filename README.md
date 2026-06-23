# Khmer Grammar Checking (Binary Classification)

ITM 454 - Natural Language Processing course project. A Khmer Grammar Checking System formulated as a binary sentence-level classification problem: given a Khmer sentence, the model predicts whether it is grammatically **acceptable (Right)** or **unacceptable (Wrong)**.

The system follows a standard supervised NLP pipeline adapted for low-resource language settings. Text is tokenized and POS-tagged with [khmer-pos-roberta](https://huggingface.co/seanghay/khmer-pos-roberta). Features are manually engineered from the POS tags and FastText embeddings, then fed into a **BiGRU classifier** with attention pooling.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters and paths
│   ├── utils.py                # Shared helpers (data loading, seeding)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # KhmerTextDataset with embedding caching
│   │   └── pos_tagger.py       # KhmerPOSTagger (seanghay/khmer-pos-roberta)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── oov.py              # EmbeddingOOVCalculator
│   │   ├── grammar.py          # SimplePOSGrammarExtractor
│   │   ├── interaction.py      # InteractionFeatureExtractor

│   │   └── semantic.py         # SemanticCoherence
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gru.py              # GRUClassifier, ImprovedGRUClassifier, AttentionPooling
│   │   ├── train_utils.py      # Training loops, evaluation loops, EarlyStopping
│   │   └── predictor.py        # SentencePredictor for inference
│   └── pipeline/
│       ├── __init__.py
│       └── feature_pipeline.py # FeaturePipeline orchestrator
├── train.py                    # Full training script
├── evaluate.py                 # Evaluation on test set
├── predict.py                  # Single-sentence prediction
├── requirements.txt
├── train_data.csv              # Pre-extracted feature CSV
├── README.md
├── Khmer_Text_Eval_BiGRU.ipynb       # Original notebook (BiGRU training)
└── Khmer_Text_Eval_Pipeline.ipynb    # Original notebook (feature pipeline)
```

## Features

| Feature | Description |
|---|---|
| `oov_ratio` | Out-of-Vocabulary ratio using FastText vocabulary |
| `dep_grammar_score` | POS-sequence grammar score (0-1) |
| `has_complete_clause` | Whether the sentence has a complete clause (0/1) |

| `semantic_coherence` | Ensemble cosine-similarity score across adjacent/content/distant words |
| `grammar_oov_interaction` | `grammar_score * (1 - oov_ratio)` |

## Model

**BiGRU with Attention Pooling**
- 300-dim FastText embeddings (cc.km.300.bin)
- 2-layer bidirectional GRU (hidden dim 256)
- Attention + mean pooling concatenation
- 3-layer MLP classifier with LayerNorm and dropout (0.5)
- Early stopping (patience 3) and gradient clipping
- Adam optimizer (lr=1e-3, weight decay=1e-5)

### Results (Test Set)

```
Accuracy:  88.79%
Precision: 85.01%
Recall:    94.18%
F1-Score:  89.36%
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py

# Evaluate
python evaluate.py

# Predict a single sentence
python predict.py "អ៊ីតាលីបានឈ្នះលើព័រទុយហ្គាល់ 31-5"
```

## Data

The training data (`train_data.csv`) contains engineered features extracted from Khmer sentences sourced from the ALT Parallel Corpus. Each row represents a sentence with numeric feature columns and a binary `sentence_correct` label.

## Requirements

- PyTorch >= 2.0
- transformers (for khmer-pos-roberta)
- fasttext (for cc.km.300.bin embeddings)
- scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm
