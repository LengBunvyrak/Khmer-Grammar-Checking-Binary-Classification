import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_gru_epoch(model, dataloader, optimizer, criterion, device, clip_value=1.0):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for embeddings, labels, lengths in tqdm(dataloader, desc="Training"):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad()
        predictions = model(embeddings, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader), correct / total


def evaluate_gru(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels, lengths in tqdm(dataloader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            predictions = model(embeddings, lengths)
            loss = criterion(predictions, labels)
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return epoch_loss / len(dataloader), correct / total, all_preds, all_labels


def train_gru_epoch_with_features(
    model, dataloader, feature_tensor, optimizer, criterion, device, clip_value=1.0
):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    batch_start_idx = 0
    for embeddings, labels, lengths in tqdm(dataloader, desc="Training"):
        batch_size = embeddings.size(0)
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad()
        extra_features = None
        if feature_tensor is not None:
            extra_features = feature_tensor[batch_start_idx : batch_start_idx + batch_size].to(
                device
            )
            batch_start_idx += batch_size
        if hasattr(model, "use_feature_fusion") and model.use_feature_fusion:
            predictions, _ = model(embeddings, lengths, extra_features)
        else:
            predictions = model(embeddings, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader), correct / total


def evaluate_gru_with_features(model, dataloader, feature_tensor, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    batch_start_idx = 0
    with torch.no_grad():
        for embeddings, labels, lengths in tqdm(dataloader, desc="Evaluating"):
            batch_size = embeddings.size(0)
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            extra_features = None
            if feature_tensor is not None:
                extra_features = feature_tensor[batch_start_idx : batch_start_idx + batch_size].to(
                    device
                )
                batch_start_idx += batch_size
            if hasattr(model, "use_feature_fusion") and model.use_feature_fusion:
                predictions, _ = model(embeddings, lengths, extra_features)
            else:
                predictions = model(embeddings, lengths)
            loss = criterion(predictions, labels)
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return epoch_loss / len(dataloader), correct / total, all_preds, all_labels


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, mode="min"):
        assert mode in {"min", "max"}
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.best_state_dict = None
        self.best_epoch = -1
        if self.mode == "min":
            self._is_improvement = lambda current, best: (best - current) > self.min_delta
            self._init_best = float("inf")
        else:
            self._is_improvement = lambda current, best: (current - best) > self.min_delta
            self._init_best = -float("inf")
        self.best = self._init_best

    def step(self, current_value, model=None, epoch=None):
        if self.best is None:
            self.best = current_value
            if model is not None:
                self.best_state_dict = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            self.best_epoch = epoch if epoch is not None else 0
            return False
        if self._is_improvement(current_value, self.best):
            self.best = current_value
            self.num_bad_epochs = 0
            self.best_epoch = epoch if epoch is not None else self.best_epoch
            if model is not None:
                self.best_state_dict = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.early_stop = True
        return self.early_stop
