"""
LSTM Autoencoder for Unsupervised Anomaly Detection.
Trained on healthy baseline operating windows to detect developing mechanical anomalies via reconstruction error.
Matches architecture from Capstone Document Section 11.3:
  Encoder: LSTM(64) -> LSTM(32)
  Bottleneck: RepeatVector(time_steps)
  Decoder: LSTM(32) -> LSTM(64)
  Output: TimeDistributed(Dense(channels))
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import torch
import torch.nn as nn
import numpy as np


class LSTMAutoencoder(nn.Module):
    """
    PyTorch implementation of the 2-stage LSTM Autoencoder for temporal sequence reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        enc_hidden_1: int = 64,
        latent_dim: int = 32,
        dec_hidden_1: int = 32,
        dec_hidden_2: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder: LSTM(64) -> LSTM(32)
        self.enc_lstm1 = nn.LSTM(
            input_size=in_channels,
            hidden_size=enc_hidden_1,
            batch_first=True,
        )
        self.enc_lstm2 = nn.LSTM(
            input_size=enc_hidden_1,
            hidden_size=latent_dim,
            batch_first=True,
        )

        # Decoder: LSTM(32) -> LSTM(64)
        self.dec_lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=dec_hidden_1,
            batch_first=True,
        )
        self.dec_lstm2 = nn.LSTM(
            input_size=dec_hidden_1,
            hidden_size=dec_hidden_2,
            batch_first=True,
        )

        # Output projection back to in_channels
        self.output_layer = nn.Linear(dec_hidden_2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, in_channels)
        Returns:
            x_recon: Reconstructed tensor of shape (batch_size, seq_len, in_channels)
        """
        batch_size, seq_len, _ = x.shape

        # Encoder
        out1, _ = self.enc_lstm1(x)                   # (B, T, 64)
        _, (h_n, _) = self.enc_lstm2(out1)            # h_n shape: (1, B, 32)
        latent = h_n.squeeze(0)                       # (B, 32)

        # Bottleneck: Repeat latent vector across all time steps
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, 32)

        # Decoder
        dec_out1, _ = self.dec_lstm1(repeated)        # (B, T, 32)
        dec_out2, _ = self.dec_lstm2(dec_out1)        # (B, T, 64)

        # Linear projection
        x_recon = self.output_layer(dec_out2)         # (B, T, in_channels)
        return x_recon


class LSTMAutoencoderDetector:
    """
    High-level wrapper for training, thresholding, and anomaly scoring using LSTM Autoencoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        threshold_factor: float = 3.0,
        device: Optional[str] = None,
    ):
        self.in_channels = in_channels
        self.threshold_factor = threshold_factor
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LSTMAutoencoder(in_channels=in_channels).to(self.device)
        self.reconstruction_threshold: float = 0.05
        self.mean_healthy_error: float = 0.01
        self.std_healthy_error: float = 0.005
        self.is_fitted: bool = False

    def fit(
        self,
        healthy_windows: np.ndarray,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 0.003,
        verbose: bool = False,
    ) -> "LSTMAutoencoderDetector":
        """
        Train autoencoder exclusively on healthy operational windows.
        Args:
            healthy_windows: Array of shape (num_windows, seq_len, in_channels)
        """
        self.model.train()
        tensor_data = torch.tensor(healthy_windows, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            avg_loss = total_loss / len(tensor_data)
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"[LSTM-AE] Epoch {epoch+1}/{epochs} - Reconstruction Loss: {avg_loss:.6f}")

        # Compute calibration threshold on healthy data: mean + 3 * std
        self.model.eval()
        with torch.no_grad():
            recon_all = self.model(tensor_data)
            errors = torch.mean((recon_all - tensor_data) ** 2, dim=(1, 2)).cpu().numpy()

        self.mean_healthy_error = float(np.mean(errors))
        self.std_healthy_error = float(np.std(errors))
        self.reconstruction_threshold = float(self.mean_healthy_error + self.threshold_factor * self.std_healthy_error)
        self.is_fitted = True
        return self

    def compute_reconstruction_error(self, windows: np.ndarray) -> np.ndarray:
        """
        Compute MSE reconstruction error for a batch of windows.
        Args:
            windows: Array of shape (num_windows, seq_len, in_channels) or (seq_len, in_channels)
        """
        if windows.ndim == 2:
            windows = np.expand_dims(windows, axis=0)

        self.model.eval()
        tensor_x = torch.tensor(windows, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(tensor_x)
            mse_per_window = torch.mean((recon - tensor_x) ** 2, dim=(1, 2)).cpu().numpy()
        return mse_per_window

    def predict_anomaly(self, windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            is_anomaly: boolean array
            normalized_anomaly_score: float array in [0.0, 1.0]
        """
        errors = self.compute_reconstruction_error(windows)
        is_anomaly = errors > self.reconstruction_threshold
        # Normalized score: reaches 1.0 when error is twice threshold
        normalized_scores = np.clip(errors / (self.reconstruction_threshold * 1.5 + 1e-9), 0.0, 1.0)
        return is_anomaly, normalized_scores

    def score_single_window(self, window: np.ndarray) -> Dict[str, Any]:
        """Score a single window array (seq_len, channels)."""
        is_anom, scores = self.predict_anomaly(window)
        errors = self.compute_reconstruction_error(window)
        return {
            "is_anomaly": bool(is_anom[0]),
            "reconstruction_error": float(np.round(errors[0], 5)),
            "anomaly_score": float(np.round(scores[0], 4)),
            "threshold": float(np.round(self.reconstruction_threshold, 5)),
        }

    def save(self, filepath: str) -> None:
        """Save model weights and calibration threshold to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.in_channels,
            "threshold": self.reconstruction_threshold,
            "mean_error": self.mean_healthy_error,
            "std_error": self.std_healthy_error,
            "is_fitted": self.is_fitted,
        }, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "LSTMAutoencoderDetector":
        """Load detector from saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        detector = cls(in_channels=checkpoint["in_channels"], device=device)
        detector.model.load_state_dict(checkpoint["model_state_dict"])
        detector.reconstruction_threshold = checkpoint["threshold"]
        detector.mean_healthy_error = checkpoint["mean_error"]
        detector.std_healthy_error = checkpoint["std_error"]
        detector.is_fitted = checkpoint["is_fitted"]
        detector.model.eval()
        return detector
