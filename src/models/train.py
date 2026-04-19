"""Módulo de treinamento da MLP."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

SEED = 42


class ChurnDataset(Dataset):
    """Dataset customizado para dados de churn."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 0.001,
    patience: int = 10,
) -> dict:
    """Treina o modelo com early stopping."""
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item()

        avg_train = epoch_train_loss / len(train_loader)
        avg_val = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %3d | Train: %.4f | Val: %.4f | Patience: %d/%d",
                epoch + 1, avg_train, avg_val, patience_counter, patience,
            )

        if patience_counter >= patience:
            logger.info("Early stopping na epoch %d", epoch + 1)
            break

    model.load_state_dict(best_model_state)
    return {"train_losses": train_losses, "val_losses": val_losses, "best_val_loss": best_val_loss}
