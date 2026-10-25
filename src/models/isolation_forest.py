"""
Isolation Forest Baseline Anomaly Detector.
Unsupervised tree-based anomaly detection using engineered tabular features.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestBaseline:
    """
    Unsupervised baseline anomaly detector.
    Operates on tabular engineered features to flag deviations from healthy baseline.
    """

    DEFAULT_FEATURE_COLS = [
        "rad_rms",
        "rad_peak",
        "rad_crest_factor",
        "rad_kurtosis",
        "rad_dominant_freq_hz",
        "rad_spectral_centroid",
        "rad_spectral_spread",
        "rad_spectral_entropy",
        "rad_band_1x_energy",
        "rad_band_2x_energy",
        "rad_band_3x_energy",
        "rad_band_bpfo_energy",
        "axial_to_radial_rms_ratio",
        "temp_mean",
        "temp_slope",
        "current_mean",
        "current_100hz_ripple",
    ]

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def _prepare_matrix(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """Extract and clean selected feature columns into a 2D numpy array."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        available_cols = [c for c in self.feature_cols if c in df.columns]
        if len(available_cols) == 0:
            raise ValueError(f"None of the required feature columns {self.feature_cols} found in input data.")

        matrix = df[available_cols].to_numpy(dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        return matrix

    def fit(self, data: pd.DataFrame) -> "IsolationForestBaseline":
        """
        Fit scaler and Isolation Forest model on reference data (preferably healthy operation).
        """
        X = self._prepare_matrix(data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict_anomaly(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly label and continuous anomaly score.
        
        Returns:
          is_anomaly: boolean array (True if anomaly, False if normal)
          anomaly_score: float array normalized to [0.0, 1.0] where 1.0 is highest anomaly
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestBaseline is not fitted. Call fit() first.")

        X = self._prepare_matrix(data)
        X_scaled = self.scaler.transform(X)

        # raw decision function: negative values are anomalies, positive are normal
        raw_scores = self.model.decision_function(X_scaled)
        # raw prediction: -1 for anomaly, 1 for inlier
        raw_preds = self.model.predict(X_scaled)

        # Normalize score into [0, 1] range: higher score = higher anomaly likelihood
        # Decision function is typically in range [-0.5, 0.5]
        normalized_scores = 1.0 / (1.0 + np.exp(raw_scores * 6.0))
        is_anomaly = raw_preds == -1

        return is_anomaly, normalized_scores

    def score_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single feature dictionary."""
        is_anom, scores = self.predict_anomaly(features)
        return {
            "is_anomaly": bool(is_anom[0]),
            "anomaly_score": float(np.round(scores[0], 4)),
        }

    def save(self, filepath: str) -> None:
        """Serialize fitted model and scaler to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "model": self.model,
            "feature_cols": self.feature_cols,
            "is_fitted": self.is_fitted,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsolationForestBaseline":
        """Load fitted model and scaler from disk."""
        data = joblib.load(filepath)
        instance = cls(feature_cols=data["feature_cols"])
        instance.scaler = data["scaler"]
        instance.model = data["model"]
        instance.is_fitted = data["is_fitted"]
        return instance
