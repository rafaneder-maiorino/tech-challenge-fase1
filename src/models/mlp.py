"""Módulo da rede neural MLP para previsão de churn."""

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """Multi-Layer Perceptron para classificação binária de churn."""

    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x).squeeze()
