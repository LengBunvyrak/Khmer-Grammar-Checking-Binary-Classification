import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    DEVICE,
    DATA_PATH,
    MODEL_SAVE_PATH,
    FEATURE_COLUMNS,
)
from src.utils import load_and_split_data, create_dataloaders
from src.data.dataset import KhmerTextDataset
from src.models.gru import GRUClassifier
from src.models.train_utils import evaluate_gru
import fasttext


def main():
    print(f"Using device: {DEVICE}")

    print("Loading FastText embedding model...")
    embedding_model = fasttext.load_model("cc.km.300.bin")

    print("\nLoading and splitting data...")
    df, X_train, X_val, X_test, y_train, y_val, y_test, scaler, _, _, _ = load_and_split_data(
        DATA_PATH, FEATURE_COLUMNS
    )

    print("\nCreating test dataset...")
    test_dataset = KhmerTextDataset(
        df.loc[X_test.index, "tokens"].tolist() if "tokens" in df.columns else df.loc[X_test.index, "text"].tolist(),
        y_test.values,
        embedding_model,
    )
    _, _, test_loader = create_dataloaders(None, None, test_dataset, BATCH_SIZE)

    print("\nLoading saved model...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
    model_state = checkpoint["model_state_dict"]

    gru_model = GRUClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    gru_model.load_state_dict(model_state)
    criterion = nn.CrossEntropyLoss()

    print("=" * 80)
    print("Evaluating GRU on Test Set")
    print("=" * 80)

    test_loss, test_acc, y_pred, y_true = evaluate_gru(
        gru_model, test_loader, criterion, DEVICE
    )

    test_precision = precision_score(y_true, y_pred)
    test_recall = recall_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)

    print(f"\nGRU Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Right", "Wrong"]
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Right", "Wrong"],
        yticklabels=["Right", "Wrong"],
    )
    plt.title("Confusion Matrix - GRU")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("gru_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
