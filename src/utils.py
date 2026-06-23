import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import FEATURE_COLUMNS, BATCH_SIZE, DEVICE
from src.data.dataset import KhmerTextDataset, collate_batch


def load_and_split_data(df_path="train_data.csv", feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    df = pd.read_csv(df_path, encoding="utf-8-sig")
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    y = df["sentence_correct"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    return df, X_train, X_val, X_test, y_train, y_val, y_test, scaler, X_train_scaled, X_val_scaled, X_test_scaled


def build_feature_tensor(df, indices, feature_columns, scaler=None):
    features = df.loc[indices, feature_columns].values
    if scaler is not None:
        features = scaler.transform(features)
    return torch.FloatTensor(features)


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    return train_loader, val_loader, test_loader


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
