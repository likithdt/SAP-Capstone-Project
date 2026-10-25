"""
CNN-LSTM Regressor for Remaining Useful Life (RUL) Prediction.
Models degradation sequences to project operational operating hours remaining before critical failure.
Matches architecture from Capstone Document Section 11.5:
  Input: (batch_size, in_channels, seq_len)
  Conv1D(64, kernel_size=3, relu) -> MaxPool
  Conv1D(128, kernel_size=3, relu) -> MaxPool
  Permute to (batch_size, time_steps, 128)
  LSTM(64, return_sequences=True)
  LSTM(32)
  Dense(32, relu) -> Dropout(0.2)
  Dense(1, linear) -> RUL (hours)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import torch
import torch.nn as nn
import numpy as np


class CNNLSTMRegressor(nn.Module):
    """
    Hybrid CNN-LSTM network for Remaining Useful Life (RUL) degradation regression.
    """

    def __init__(self, in_channels: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.in_channels = in_channels

        # Conv Block 1: Conv1D(64) -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Conv Block 2: Conv1D(128) -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Recurrent temporal degradation modeling: LSTM(64) -> LSTM(32)
        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True,
        )

        # Regression Head: Dense(32) -> Dropout(0.2) -> Dense(1)
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, in_channels, seq_len)
        Returns:
            rul_hours: Tensor of shape (batch_size, 1)
        """
        # CNN pattern extraction
        c1 = self.conv1(x)    # (B, 64, T/2)
        c2 = self.conv2(c1)   # (B, 128, T/4)

        # Permute for LSTM sequence input: (B, T/4, 128)
        lstm_in = c2.permute(0, 2, 1)

        # LSTM sequence modeling
        out1, _ = self.lstm1(lstm_in)          # (B, T/4, 64)
        _, (h_n, _) = self.lstm2(out1)         # h_n shape: (1, B, 32)
        last_hidden = h_n.squeeze(0)          # (B, 32)

        # RUL scalar regression
        rul = self.head(last_hidden)          # (B, 1)
        return rul


class RULPredictor:
    """
    High-level training and inference wrapper for the CNN-LSTM RUL regression model.
    """

    def __init__(self, in_channels: int = 3, max_rul_hours: float = 120.0, device: Optional[str] = None):
        self.in_channels = in_channels
        self.max_rul_hours = max_rul_hours
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CNNLSTMRegressor(in_channels=in_channels).to(self.device)
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y_rul: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> "RULPredictor":
        """
        Train regression model on sequence data and true RUL hours.
        Args:
            X: Array of shape (N, in_channels, seq_len) or (N, seq_len, in_channels)
            y_rul: Float array of true remaining hours (N,)
        """
        if X.ndim == 3 and X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        self.model.train()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        tensor_y = torch.tensor(y_rul, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()  # Robust Huber loss for regression

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred_rul = self.model(batch_x)
                loss = criterion(pred_rul, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                avg_loss = total_loss / len(tensor_x)
                print(f"[CNN-LSTM RUL] Epoch {epoch+1}/{epochs} - SmoothL1 Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict Remaining Useful Life in operating hours.
        Returns:
            rul_hours: 1D float array (N,) bounded between [0.0, max_rul_hours]
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        if X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor_x).cpu().numpy().squeeze(axis=1)

        # Clip RUL to physical non-negative range
        pred = np.clip(pred, 0.0, self.max_rul_hours)
        return pred

    def predict_single(self, window: np.ndarray) -> float:
        """Predict RUL for a single sequence window."""
        rul = self.predict(window)
        return float(np.round(rul[0], 1))

    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.in_channels,
            "max_rul_hours": self.max_rul_hours,
            "is_fitted": self.is_fitted,
        }, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "RULPredictor":
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        predictor = cls(
            in_channels=checkpoint["in_channels"],
            max_rul_hours=checkpoint["max_rul_hours"],
            device=device,
        )
        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        predictor.is_fitted = checkpoint["is_fitted"]
        predictor.model.eval()
        return predictor
