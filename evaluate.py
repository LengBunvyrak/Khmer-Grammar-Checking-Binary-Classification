import torch
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
    USE_FEATURE_FUSION,
    NUM_EXTRA_FEATURES,
    POS_TAGS,
    POS_EMBEDDING_DIM,
)
from src.utils import load_and_split_data, create_dataloaders, build_feature_tensor
from src.data.dataset import KhmerTextDataset
from src.models.gru import ImprovedGRUClassifier
from src.models.train_utils import predict_probs_with_features


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
        df.loc[X_test.index, "pos_tags"].tolist() if "pos_tags" in df.columns else [[] for _ in range(len(X_test))],
        y_test.values,
        embedding_model,
    )
    _, _, test_loader = create_dataloaders(None, None, test_dataset, BATCH_SIZE)

    print("\nLoading saved model...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model_state = checkpoint["model_state_dict"]
    scaler = checkpoint.get("scaler", None)
    threshold = checkpoint.get("threshold", 0.5)

    gru_model = ImprovedGRUClassifier(
        EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT,
        num_extra_features=NUM_EXTRA_FEATURES, use_feature_fusion=USE_FEATURE_FUSION,
        pos_vocab_size=len(POS_TAGS), pos_embedding_dim=POS_EMBEDDING_DIM,
    ).to(DEVICE)
    gru_model.load_state_dict(model_state)

    test_feat_tensor = build_feature_tensor(df, X_test.index, FEATURE_COLUMNS, scaler)

    print("=" * 80)
    print("Evaluating Improved GRU on Test Set (with feature fusion)")
    print(f"Decision threshold: P(Right) >= {threshold:.3f}")
    print("=" * 80)

    test_probs, y_true = predict_probs_with_features(
        gru_model, test_loader, test_feat_tensor, DEVICE
    )
    y_pred = (test_probs[:, 1] >= threshold).astype(int)

    test_acc = accuracy_score(y_true, y_pred)
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
            y_true, y_pred, target_names=["Wrong", "Right"]
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Wrong", "Right"],
        yticklabels=["Wrong", "Right"],
    )
    plt.title("Confusion Matrix - GRU")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("gru_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
