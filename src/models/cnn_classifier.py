"""
1D-CNN Fault Classifier for Multi-Class Failure Mode Diagnosis.
Categorizes asset state into 7 operational classes matching Capstone Section 11.4:
  Input: (batch_size, in_channels, seq_len)
  Conv1D(32, kernel_size=3, relu) -> BatchNorm -> MaxPool
  Conv1D(64, kernel_size=3, relu) -> BatchNorm -> MaxPool
  Conv1D(128, kernel_size=3, relu) -> GlobalAveragePooling
  Dense(64, relu) -> Dropout(0.3)
  Dense(num_classes=7, softmax)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import FaultClass


class FaultClassifier1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for multi-class rotating machinery fault classification.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 7, dropout_rate: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Layer 1: Conv1D(32) -> BatchNorm -> ReLU -> MaxPool
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 2: Conv1D(64) -> BatchNorm -> ReLU -> MaxPool
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 3: Conv1D(128) -> BatchNorm -> ReLU -> GlobalAveragePooling
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier head: Dense(64) -> Dropout(0.3) -> Dense(num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, in_channels, seq_len)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits


class FaultClassifier:
    """
    High-level training and inference interface for the 1D-CNN fault classification model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        device: Optional[str] = None,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FaultClassifier1DCNN(in_channels=in_channels, num_classes=num_classes).to(self.device)
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> "FaultClassifier":
        """
        Train 1D-CNN classifier on labeled vibration sequences.
        Args:
            X: Array of shape (N, in_channels, seq_len) or (N, seq_len, in_channels)
            y: Integer class labels of shape (N,)
        """
        # Ensure input is (N, channels, seq_len)
        if X.ndim == 3 and X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        self.model.train()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        tensor_y = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()

            train_acc = correct / len(tensor_x)
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"[1D-CNN] Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(tensor_x):.4f} - Acc: {train_acc*100:.1f}%")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict class IDs, confidence scores, and full probability distributions.
        Returns:
            predicted_classes: int array (N,)
            confidences: float array (N,) in [0.0, 1.0]
            probabilities: float array (N, num_classes)
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        # Ensure (N, channels, seq_len)
        if X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        predicted_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return predicted_classes, confidences, probs

    def predict_single(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Predict diagnostics for a single window array.
        """
        classes, confs, probs = self.predict(window)
        pred_class = int(classes[0])
        fault_enum = FaultClass(pred_class)
        return {
            "predicted_class": pred_class,
            "predicted_label": fault_enum.label,
            "confidence": float(np.round(confs[0], 4)),
            "class_probabilities": {
                FaultClass(c).label: float(np.round(probs[0, c], 4))
                for c in range(self.num_classes)
            },
        }

    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "is_fitted": self.is_fitted,
        }, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "FaultClassifier":
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        classifier = cls(
            in_channels=checkpoint["in_channels"],
            num_classes=checkpoint["num_classes"],
            device=device,
        )
        classifier.model.load_state_dict(checkpoint["model_state_dict"])
        classifier.is_fitted = checkpoint["is_fitted"]
        classifier.model.eval()
        return classifier
