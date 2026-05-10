from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


ClassifierKind = Literal["multi_task", "mlp", "lstm"]


class MultiTaskHead(nn.Module):
    """62-D kinematic feature input with classification and quality regression heads."""

    def __init__(self, input_dim: int = 62, num_classes: int = 6, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        score = torch.sigmoid(self.regressor(encoded)).squeeze(-1) * 100.0
        return logits, score

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class MultiTaskLoss(nn.Module):
    """Joint loss: CE(class) + 0.5 * MSE(quality)."""

    def __init__(self, regression_weight: float = 0.5) -> None:
        super().__init__()
        self.regression_weight = regression_weight
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        logits: torch.Tensor,
        scores: torch.Tensor,
        class_labels: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        return self.ce(logits, class_labels) + self.regression_weight * self.mse(scores, quality_scores)


class FeatureMLPClassifier(nn.Module):
    """Classification-only MLP for YOLOv8-Pose + kinematic features ablation."""

    def __init__(self, input_dim: int = 62, num_classes: int = 6, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SequenceLSTMClassifier(nn.Module):
    """LSTM baseline over per-frame 15-D kinematic features."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


def build_classifier_model(kind: ClassifierKind, num_classes: int = 6, dropout: float = 0.3) -> nn.Module:
    if kind == "multi_task":
        return MultiTaskHead(num_classes=num_classes, dropout=dropout)
    if kind == "mlp":
        return FeatureMLPClassifier(num_classes=num_classes, dropout=dropout)
    if kind == "lstm":
        return SequenceLSTMClassifier(num_classes=num_classes, dropout=dropout)
    raise ValueError("kind must be one of: multi_task, mlp, lstm")

