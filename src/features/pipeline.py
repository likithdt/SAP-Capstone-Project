"""
Comprehensive Feature Extraction Pipeline for Industrial Telemetry.
Combines time-domain statistics, FFT spectral analysis, envelope demodulation,
cross-channel axial-radial ratios, thermal trends, electrical ripple, and sensor integrity metrics.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

from src.config import AssetConfig, ClassThresholds, DEFAULT_ASSET_CONFIG
from src.features.time_domain import (
    compute_time_domain_features,
    compute_temperature_features,
    compute_current_features,
)
from src.features.frequency_domain import compute_frequency_domain_features


class FeatureExtractionPipeline:
    """
    Production-grade Feature Engineering Pipeline.
    Packages time-domain and frequency-domain feature extraction into a reusable,
    vectorized pipeline for downstream ML model training and real-time edge/cloud inference.
    """

    def __init__(self, config: Optional[AssetConfig] = None):
        self.config = config or DEFAULT_ASSET_CONFIG
        self.fs = self.config.sampling_rate_hz
        self.f0 = self.config.fundamental_freq_hz
        self.bpfo = self.config.bpfo_freq_hz

    def extract_from_arrays(
        self,
        accel_x: np.ndarray,
        accel_y: Optional[np.ndarray] = None,
        accel_z: Optional[np.ndarray] = None,
        temperature_c: Optional[np.ndarray] = None,
        current_a: Optional[np.ndarray] = None,
        speed_rpm: Optional[np.ndarray] = None,
        fault_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract complete tabular feature vector from raw sensor channel arrays.
        """
        features: Dict[str, Any] = {}

        # ---------------------------------------------------------------------
        # 1. Sensor Data Quality & Anomaly Integrity Checks
        # ---------------------------------------------------------------------
        total_samples = len(accel_x)
        nan_count = int(np.isnan(accel_x).sum())
        if temperature_c is not None:
            nan_count += int(np.isnan(temperature_c).sum())
        if current_a is not None:
            nan_count += int(np.isnan(current_a).sum())

        clean_x = accel_x[~np.isnan(accel_x)]
        dropout_count = int(np.sum(clean_x == 0.0))
        dropout_ratio = float(dropout_count / total_samples) if total_samples > 0 else 0.0
        extreme_spikes = int(np.sum(np.abs(clean_x) > ClassThresholds.SENSOR_SPIKE_THRESHOLD_G))

        features["integrity_nan_count"] = nan_count
        features["integrity_dropout_ratio"] = dropout_ratio
        features["integrity_extreme_spike_count"] = extreme_spikes
        features["integrity_has_sensor_glitch"] = int(nan_count > 0 or dropout_ratio > 0.05 or extreme_spikes > 0)

        # ---------------------------------------------------------------------
        # 2. Radial Vibration (accel_x): Time-Domain & Frequency-Domain
        # ---------------------------------------------------------------------
        time_feats_x = compute_time_domain_features(clean_x, prefix="rad")
        features.update(time_feats_x)

        freq_feats_x = compute_frequency_domain_features(
            clean_x,
            fs=self.fs,
            f0=self.f0,
            bpfo=self.bpfo,
            prefix="rad"
        )
        features.update(freq_feats_x)

        # ---------------------------------------------------------------------
        # 3. Tangential & Axial Vibration (accel_y, accel_z)
        # ---------------------------------------------------------------------
        if accel_y is not None:
            clean_y = accel_y[~np.isnan(accel_y)]
            time_feats_y = compute_time_domain_features(clean_y, prefix="tan")
            features[f"tan_rms"] = time_feats_y["tan_rms"]
            features[f"tan_peak"] = time_feats_y["tan_peak"]
            features[f"tan_kurtosis"] = time_feats_y["tan_kurtosis"]

        if accel_z is not None:
            clean_z = accel_z[~np.isnan(accel_z)]
            time_feats_z = compute_time_domain_features(clean_z, prefix="ax")
            features[f"ax_rms"] = time_feats_z["ax_rms"]
            features[f"ax_peak"] = time_feats_z["ax_peak"]
            features[f"ax_kurtosis"] = time_feats_z["ax_kurtosis"]

            # Multi-channel Cross Ratio: Axial-to-Radial ratio (crucial for Misalignment!)
            rad_rms = features["rad_rms"]
            ax_rms = features["ax_rms"]
            features["axial_to_radial_rms_ratio"] = float(ax_rms / rad_rms) if rad_rms > 1e-6 else 0.0

            # Triaxial vector magnitude: |a| = sqrt(ax^2 + ay^2 + az^2)
            if accel_y is not None and len(clean_x) == len(clean_y) == len(clean_z):
                mag = np.sqrt(clean_x**2 + clean_y**2 + clean_z**2)
                features["triaxial_mag_rms"] = float(np.sqrt(np.mean(mag**2)))
                features["triaxial_mag_peak"] = float(np.max(mag))

        # ---------------------------------------------------------------------
        # 4. Thermal Features
        # ---------------------------------------------------------------------
        if temperature_c is not None:
            temp_feats = compute_temperature_features(temperature_c)
            features.update(temp_feats)
            # Critical temperature alert flag
            features["temp_critical_flag"] = int(temp_feats["temp_max"] >= ClassThresholds.THERMAL_OVERHEATING_CRITICAL_C)

        # ---------------------------------------------------------------------
        # 5. Electrical Current Features
        # ---------------------------------------------------------------------
        if current_a is not None:
            curr_feats = compute_current_features(current_a, fs=self.fs)
            features.update(curr_feats)
            # Motor overload current flag
            features["current_overload_flag"] = int(curr_feats["current_mean"] >= ClassThresholds.OVERLOAD_CURRENT_MIN_A)

        # ---------------------------------------------------------------------
        # 6. Rotational Speed (RPM)
        # ---------------------------------------------------------------------
        if speed_rpm is not None:
            clean_rpm = speed_rpm[~np.isnan(speed_rpm)]
            features["speed_rpm_mean"] = float(np.mean(clean_rpm)) if len(clean_rpm) > 0 else self.config.nominal_speed_rpm
            features["speed_rpm_std"] = float(np.std(clean_rpm)) if len(clean_rpm) > 0 else 0.0
        else:
            features["speed_rpm_mean"] = self.config.nominal_speed_rpm
            features["speed_rpm_std"] = 0.0

        if fault_class is not None:
            features["fault_class"] = int(fault_class)

        return features

    def extract_from_batch(self, batch: Any) -> Dict[str, Any]:
        """Extract features directly from a TelemetryBatch instance."""
        return self.extract_from_arrays(
            accel_x=batch.accel_x,
            accel_y=batch.accel_y,
            accel_z=batch.accel_z,
            temperature_c=batch.temperature_c,
            current_a=batch.current_a,
            speed_rpm=batch.speed_rpm,
            fault_class=getattr(batch, "fault_class", None),
        )

    def extract_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from a window DataFrame."""
        return self.extract_from_arrays(
            accel_x=df["accel_x"].to_numpy(),
            accel_y=df["accel_y"].to_numpy() if "accel_y" in df.columns else None,
            accel_z=df["accel_z"].to_numpy() if "accel_z" in df.columns else None,
            temperature_c=df["temperature_c"].to_numpy() if "temperature_c" in df.columns else None,
            current_a=df["current_a"].to_numpy() if "current_a" in df.columns else None,
            speed_rpm=df["speed_rpm"].to_numpy() if "speed_rpm" in df.columns else None,
            fault_class=int(df["fault_class"].iloc[0]) if "fault_class" in df.columns else None,
        )

    def transform_batches(self, batches: List[Any]) -> pd.DataFrame:
        """
        Transform a list of TelemetryBatch or DataFrame objects into a tabular feature matrix.
        Returns a clean pandas DataFrame ready for ML pipelines.
        """
        rows = []
        for b in batches:
            if hasattr(b, "accel_x"):
                feats = self.extract_from_batch(b)
            elif isinstance(b, pd.DataFrame):
                feats = self.extract_from_dataframe(b)
            elif isinstance(b, dict):
                feats = self.extract_from_arrays(**b)
            else:
                raise TypeError(f"Unsupported batch item type: {type(b)}")
            rows.append(feats)
        return pd.DataFrame(rows)

    def build_feature_dataset(
        self,
        samples_per_class: int = 100,
        duration_sec: float = 1.0,
        seed: int = 42,
        output_dir: str = "data/processed",
    ) -> pd.DataFrame:
        """
        Generate synthetic telemetry across all 7 classes and extract complete engineered features.
        Saves resulting dataset as CSV and Parquet in output_dir.
        """
        import os
        from src.digital_twin.simulator import PhysicalSignalSimulator
        from src.config import FaultClass

        simulator = PhysicalSignalSimulator(self.config)
        batches = []
        rng = np.random.default_rng(seed)

        for fault in FaultClass:
            for s in range(samples_per_class):
                speed_jitter = rng.uniform(-15.0, 15.0)
                b = simulator.generate_window(
                    fault_class=fault,
                    duration_sec=duration_sec,
                    speed_rpm=self.config.nominal_speed_rpm + speed_jitter,
                    seed=int(rng.integers(1, 1000000)),
                    allow_nan=(fault == FaultClass.SENSOR_ANOMALY and rng.random() > 0.5),
                )
                batches.append(b)

        df_features = self.transform_batches(batches)

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "engineered_features.csv")
        parquet_path = os.path.join(output_dir, "engineered_features.parquet")
        
        df_features.to_csv(csv_path, index=False)
        try:
            df_features.to_parquet(parquet_path, index=False)
        except Exception:
            pass

        return df_features


if __name__ == "__main__":
    print("=" * 80)
    print("FEATURE ENGINEERING PIPELINE: EXTRACTING COMPLETE DATASET")
    print("=" * 80)
    pipeline = FeatureExtractionPipeline()
    print(f"Sampling Rate: {pipeline.fs} Hz | f0: {pipeline.f0} Hz | BPFO: {pipeline.bpfo:.2f} Hz\n")
    print("Extracting features from 100 windows/class (700 windows total)...")
    df = pipeline.build_feature_dataset(samples_per_class=100, output_dir="data/processed")
    print(f"Extraction complete! Feature matrix shape: {df.shape}")
    print(f"Saved to: data/processed/engineered_features.csv (and .parquet)\n")
    print("Engineered feature columns summary:")
    print(f"  Time-domain features: {[c for c in df.columns if c.startswith('rad_') and ('rms' in c or 'kurt' in c or 'crest' in c)]}")
    print(f"  Frequency features:   {[c for c in df.columns if 'band' in c or 'freq' in c or 'envelope' in c][:5]}")
    print(f"  Thermal & Current:    {[c for c in df.columns if 'temp' in c or 'current' in c]}")
    print(f"  Integrity checks:     {[c for c in df.columns if 'integrity' in c]}")
    print("=" * 80)

