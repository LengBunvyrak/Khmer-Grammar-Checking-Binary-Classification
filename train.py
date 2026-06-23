import gc
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    N_EPOCHS,
    BATCH_SIZE,
    PATIENCE,
    CLIP_VALUE,
    DEVICE,
    DATA_PATH,
    MODEL_SAVE_PATH,
    FASTTEXT_MODEL_PATH,
    FEATURE_COLUMNS,
)
from src.utils import set_seed, load_and_split_data, create_dataloaders, build_feature_tensor
from src.data.dataset import KhmerTextDataset
from src.models.gru import GRUClassifier
from src.models.train_utils import train_gru_epoch, evaluate_gru, EarlyStopping
import fasttext


def main():
    set_seed()
    print(f"Using device: {DEVICE}")

    print("Loading FastText embedding model...")
    embedding_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    print("\nLoading and splitting data...")
    df, X_train, X_val, X_test, y_train, y_val, y_test, scaler, _, _, _ = load_and_split_data(
        DATA_PATH, FEATURE_COLUMNS
    )

    print("\nCreating datasets...")
    train_dataset = KhmerTextDataset(
        df.loc[X_train.index, "tokens"].tolist() if "tokens" in df.columns else df.loc[X_train.index, "text"].tolist(),
        y_train.values,
        embedding_model,
    )
    val_dataset = KhmerTextDataset(
        df.loc[X_val.index, "tokens"].tolist() if "tokens" in df.columns else df.loc[X_val.index, "text"].tolist(),
        y_val.values,
        embedding_model,
    )
    test_dataset = KhmerTextDataset(
        df.loc[X_test.index, "tokens"].tolist() if "tokens" in df.columns else df.loc[X_test.index, "text"].tolist(),
        y_test.values,
        embedding_model,
    )

    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")
    print(f"Test Dataset size: {len(test_dataset)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE
    )

    print("\nInitializing GRU model...")
    gru_model = GRUClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("=" * 80)
    print("Training GRU Model")
    print("=" * 80)
    print(f"Embedding Dim: {EMBEDDING_DIM}")
    print(f"Hidden Dim: {HIDDEN_DIM}")
    print(f"Num Layers: {N_LAYERS}")
    print(f"Dropout: {DROPOUT}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print("=" * 80)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    early_stopper = EarlyStopping(patience=PATIENCE, min_delta=0.0, mode="min")

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{N_EPOCHS}")
        print("-" * 80)
        train_loss, train_acc = train_gru_epoch(
            gru_model, train_loader, optimizer, criterion, DEVICE, CLIP_VALUE
        )
        val_loss, val_acc, _, _ = evaluate_gru(gru_model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        early_stopper.step(val_loss, gru_model, epoch=epoch + 1)
        if early_stopper.early_stop:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Restoring best weights from epoch {early_stopper.best_epoch}."
            )
            break

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Validation Accuracy (observed): {best_val_acc * 100:.2f}%")
    if early_stopper.best_epoch != -1:
        print(
            f"Best model (by val_loss) was at epoch {early_stopper.best_epoch} "
            f"with value {early_stopper.best:.4f}"
        )
    print("=" * 80)

    if early_stopper.best_state_dict is not None:
        best_model_state = early_stopper.best_state_dict
    if best_model_state is not None:
        gru_model.load_state_dict(best_model_state)
        torch.save(
            {"model_state_dict": best_model_state, "scaler": scaler, "feature_columns": FEATURE_COLUMNS},
            MODEL_SAVE_PATH,
        )
        print(f"Best model saved to {MODEL_SAVE_PATH}")
    else:
        print("Warning: No best model state captured; using final epoch weights.")
        torch.save(
            {"model_state_dict": gru_model.state_dict(), "scaler": scaler, "feature_columns": FEATURE_COLUMNS},
            MODEL_SAVE_PATH,
        )


if __name__ == "__main__":
    main()
