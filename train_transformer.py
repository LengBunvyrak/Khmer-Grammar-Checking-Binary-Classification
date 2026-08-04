"""
Fine-tune a pretrained Khmer transformer (seanghay/xlm-roberta-khmer-small)
for the binary sentence-grammar task. Replaces the FastText + BiGRU + manual
feature pipeline entirely: the transformer reads raw Khmer text directly.

Usage:
    python train_transformer.py
    python evaluate_transformer.py
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import DATA_PATH, MODEL_SAVE_PATH, CLASS_WEIGHTS, SEED

MODEL_NAME = "seanghay/xlm-roberta-khmer-small"
MAX_LENGTH = 128
BATCH_SIZE = 16
N_EPOCHS = 3
LEARNING_RATE = 2e-5
TRANSFORMER_SAVE_PATH = "transformer_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[["text", "sentence_correct"]]
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["text"], df["sentence_correct"], test_size=0.2, random_state=SEED, stratify=df["sentence_correct"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_batches(texts, labels, tokenizer):
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts.iloc[i : i + BATCH_SIZE].tolist()
        batch_labels = torch.LongTensor(labels.iloc[i : i + BATCH_SIZE].values)
        enc = tokenizer(
            batch_texts, padding="longest", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        yield enc, batch_labels


def run_epoch(model, batches, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for enc, labels in batches:
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(**enc).logits
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs[:, 1].detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item() * labels.size(0)
        correct += (probs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def find_best_threshold(probs, labels):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.5, 1.0, 0.05):
        f1 = f1_score(labels, probs >= t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Using device: {DEVICE}")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    print(f"Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(CLASS_WEIGHTS).to(DEVICE))

    best_val_f1, best_state, best_threshold = 0.0, None, 0.5
    for epoch in range(N_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(
            model, make_batches(X_train, y_train, tokenizer), optimizer, criterion, DEVICE, train=True
        )
        val_loss, val_acc, val_probs, val_labels = run_epoch(
            model, make_batches(X_val, y_val, tokenizer), optimizer, criterion, DEVICE, train=False
        )
        val_f1 = f1_score(val_labels, val_probs >= 0.5)
        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} | Train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| Val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_threshold = find_best_threshold(val_probs, val_labels)

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "tokenizer_name": MODEL_NAME,
            "threshold": best_threshold,
            "num_labels": 2,
        },
        TRANSFORMER_SAVE_PATH,
    )
    print(f"Best val F1 {best_val_f1:.4f} at threshold {best_threshold:.3f}. Saved to {TRANSFORMER_SAVE_PATH}")

    print("\nEvaluating on test set...")
    _, _, test_probs, test_labels = run_epoch(
        model, make_batches(X_test, y_test, tokenizer), optimizer, criterion, DEVICE, train=False
    )
    y_pred = (test_probs >= best_threshold).astype(int)
    print(f"Accuracy:  {accuracy_score(test_labels, y_pred):.4f}")
    print(f"Precision: {precision_score(test_labels, y_pred):.4f}")
    print(f"Recall:    {recall_score(test_labels, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(test_labels, y_pred):.4f}")
    print("\n" + classification_report(test_labels, y_pred, target_names=["Wrong", "Right"]))


if __name__ == "__main__":
    main()
