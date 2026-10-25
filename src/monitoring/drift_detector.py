"""
Evidently AI Industrial IoT Drift and Data Quality Monitoring Engine.
Monitors vibration, temperature, current features, and model diagnostic predictions
for statistical distribution shifts and triggers closed-loop automated retraining workflows.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import os
import numpy as np
import pandas as pd

# Safe Evidently imports across 0.4.x - 0.7.x+
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
except (ImportError, ModuleNotFoundError):
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.legacy.test_suite import TestSuite
    from evidently.legacy.test_preset import DataDriftTestPreset

from src.config import ClassThresholds, MLOpsConfig, DEFAULT_MLOPS_CONFIG
from src.models.isolation_forest import IsolationForestBaseline


@dataclass
class DriftResult:
    """Structured drift evaluation summary matching SAP MLOps telemetry schema."""
    timestamp: str
    dataset_drift: bool
    drift_share: float
    number_of_drifted_features: int
    total_features: int
    drifted_features: List[str]
    feature_metrics: Dict[str, Dict[str, Any]]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert drift result to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "dataset_drift": self.dataset_drift,
            "drift_share": round(self.drift_share, 4),
            "number_of_drifted_features": self.number_of_drifted_features,
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "feature_metrics": self.feature_metrics,
            "summary": self.summary,
        }


@dataclass
class DataQualityResult:
    """Integrity and sensor data hygiene evaluation."""
    timestamp: str
    has_critical_issues: bool
    nan_count: int
    infinite_count: int
    dropout_ratio: float
    spike_count: int
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "has_critical_issues": self.has_critical_issues,
            "nan_count": self.nan_count,
            "infinite_count": self.infinite_count,
            "dropout_ratio": round(self.dropout_ratio, 4),
            "spike_count": self.spike_count,
            "issues": self.issues,
        }


@dataclass
class RetrainingDecision:
    """Automated MLOps closed-loop retraining trigger decision."""
    trigger_retraining: bool
    severity: str  # "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    reason: str
    timestamp: str
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_retraining": self.trigger_retraining,
            "severity": self.severity,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metrics_snapshot": self.metrics_snapshot,
        }


class EvidentlyDriftDetector:
    """
    Evidently AI Drift Monitoring Controller.
    Performs statistical distribution checks (Kolmogorov-Smirnov, Wasserstein, Cramer's V)
    between reference baseline operation and live streaming telemetry batches.
    """

    CORE_MONITORED_FEATURES = [
        "rad_rms",
        "rad_peak",
        "rad_crest_factor",
        "rad_kurtosis",
        "rad_dominant_freq_hz",
        "rad_spectral_centroid",
        "rad_spectral_spread",
        "rad_band_1x_energy",
        "rad_band_2x_energy",
        "rad_band_bpfo_energy",
        "axial_to_radial_rms_ratio",
        "temp_mean",
        "temp_slope",
        "current_mean",
        "current_100hz_ripple",
    ]

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        drift_share_threshold: Optional[float] = None,
        p_value_threshold: Optional[float] = None,
        reports_dir: Optional[str] = None,
        config: Optional[MLOpsConfig] = None,
    ):
        self.config = config or DEFAULT_MLOPS_CONFIG
        self.feature_cols = feature_cols or self.CORE_MONITORED_FEATURES
        self.drift_share_threshold = (
            drift_share_threshold if drift_share_threshold is not None else self.config.drift_share_threshold
        )
        self.p_value_threshold = (
            p_value_threshold if p_value_threshold is not None else self.config.drift_p_value_threshold
        )
        self.reports_dir = reports_dir or self.config.report_output_dir
        os.makedirs(self.reports_dir, exist_ok=True)

        self.reference_data: Optional[pd.DataFrame] = None
        if reference_data is not None:
            self.set_reference_data(reference_data)

    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """Sanitize and set reference baseline telemetry dataset."""
        if reference_data.empty:
            raise ValueError("Reference dataset cannot be empty.")
        self.reference_data = reference_data.copy()

    def _select_numerical_features(
        self,
        df: pd.DataFrame,
        custom_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Filter dataset to validated numerical feature columns."""
        target_cols = custom_cols or self.feature_cols
        available = [c for c in target_cols if c in df.columns]
        if not available:
            # Fallback to all numerical columns
            available = list(df.select_dtypes(include=[np.number]).columns)
            # Remove metadata identifiers if present
            available = [c for c in available if c not in ["fault_class", "speed_rpm", "integrity_has_sensor_glitch"]]

        if not available:
            raise ValueError("No valid numerical features found for drift monitoring.")

        subset = df[available].copy()
        # Clean infinite/NaN values for reliable statistical calculations
        subset = subset.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return subset

    def evaluate_data_drift(
        self,
        current_data: pd.DataFrame,
        custom_features: Optional[List[str]] = None,
    ) -> DriftResult:
        """
        Evaluate feature data drift between reference baseline and current batch.
        """
        if self.reference_data is None:
            raise RuntimeError("Reference data not initialized. Call set_reference_data() first.")

        ref_clean = self._select_numerical_features(self.reference_data, custom_features)
        curr_clean = self._select_numerical_features(current_data, custom_features)

        # Match columns between ref and curr
        common_cols = [c for c in ref_clean.columns if c in curr_clean.columns]
        if not common_cols:
            raise ValueError("No common numerical feature columns between reference and current datasets.")

        ref_clean = ref_clean[common_cols]
        curr_clean = curr_clean[common_cols]

        report = Report(metrics=[DataDriftPreset(drift_share=self.drift_share_threshold)])
        report.run(reference_data=ref_clean, current_data=curr_clean)
        report_dict = report.as_dict()

        # Extract drift metrics from report results
        dataset_drift = False
        drift_share = 0.0
        number_of_drifted = 0
        total_cols = len(common_cols)
        drifted_features: List[str] = []
        feature_metrics: Dict[str, Dict[str, Any]] = {}

        for m in report_dict.get("metrics", []):
            res = m.get("result", {})
            if "dataset_drift" in res:
                dataset_drift = bool(res["dataset_drift"])
            if "share_of_drifted_columns" in res:
                drift_share = float(res["share_of_drifted_columns"])
            if "number_of_drifted_columns" in res:
                number_of_drifted = int(res["number_of_drifted_columns"])
            if "drift_by_columns" in res:
                for col, info in res["drift_by_columns"].items():
                    is_col_drift = bool(info.get("drift_detected", False))
                    p_val = float(info.get("drift_score", 1.0))
                    stat_test = str(info.get("stat_test_name", "statistical_test"))
                    feature_metrics[col] = {
                        "drift_detected": is_col_drift,
                        "drift_score": p_val,
                        "stat_test": stat_test,
                    }
                    if is_col_drift and col not in drifted_features:
                        drifted_features.append(col)

        # Synchronize exact count and share
        number_of_drifted = len(drifted_features)
        drift_share = float(number_of_drifted / total_cols) if total_cols > 0 else 0.0

        if number_of_drifted == 0:
            dataset_drift = False
        elif drift_share >= self.drift_share_threshold:
            dataset_drift = True

        timestamp = datetime.now(timezone.utc).isoformat()
        summary = (
            f"Data Drift: {'DETECTED' if dataset_drift else 'NORMAL'} | "
            f"Drift Share: {drift_share:.1%} ({number_of_drifted}/{total_cols} features) | "
            f"Threshold: {self.drift_share_threshold:.1%}"
        )

        return DriftResult(
            timestamp=timestamp,
            dataset_drift=dataset_drift,
            drift_share=drift_share,
            number_of_drifted_features=number_of_drifted,
            total_features=total_cols,
            drifted_features=drifted_features,
            feature_metrics=feature_metrics,
            summary=summary,
        )

    def evaluate_target_drift(
        self,
        reference_predictions: pd.DataFrame,
        current_predictions: pd.DataFrame,
    ) -> DriftResult:
        """
        Evaluate prediction/target drift (e.g. anomaly scores, predicted failure classes, RUL).
        """
        common_cols = [c for c in reference_predictions.columns if c in current_predictions.columns]
        if not common_cols:
            raise ValueError("No common prediction columns between reference and current predictions.")

        report = Report(metrics=[DataDriftPreset(drift_share=self.drift_share_threshold)])
        report.run(
            reference_data=reference_predictions[common_cols],
            current_data=current_predictions[common_cols],
        )
        report_dict = report.as_dict()

        dataset_drift = False
        drift_share = 0.0
        number_of_drifted = 0
        drifted_targets: List[str] = []
        feature_metrics: Dict[str, Dict[str, Any]] = {}

        for m in report_dict.get("metrics", []):
            res = m.get("result", {})
            if "dataset_drift" in res:
                dataset_drift = bool(res["dataset_drift"])
            if "share_of_drifted_columns" in res:
                drift_share = float(res["share_of_drifted_columns"])
            if "number_of_drifted_columns" in res:
                number_of_drifted = int(res["number_of_drifted_columns"])
            if "drift_by_columns" in res:
                for col, info in res["drift_by_columns"].items():
                    is_drift = bool(info.get("drift_detected", False))
                    p_val = float(info.get("drift_score", 1.0))
                    feature_metrics[col] = {
                        "drift_detected": is_drift,
                        "drift_score": p_val,
                    }
                    if is_drift and col not in drifted_targets:
                        drifted_targets.append(col)

        # Synchronize exact count and share
        total_targets = len(common_cols)
        number_of_drifted = len(drifted_targets)
        drift_share = float(number_of_drifted / total_targets) if total_targets > 0 else 0.0

        if number_of_drifted == 0:
            dataset_drift = False
        elif drift_share >= self.drift_share_threshold:
            dataset_drift = True

        timestamp = datetime.now(timezone.utc).isoformat()
        summary = (
            f"Target/Prediction Drift: {'DETECTED' if dataset_drift else 'NORMAL'} | "
            f"Drifted targets: {drifted_targets}"
        )

        return DriftResult(
            timestamp=timestamp,
            dataset_drift=dataset_drift,
            drift_share=drift_share,
            number_of_drifted_features=number_of_drifted,
            total_features=len(common_cols),
            drifted_features=drifted_targets,
            feature_metrics=feature_metrics,
            summary=summary,
        )

    def evaluate_data_quality(self, current_data: pd.DataFrame) -> DataQualityResult:
        """
        Assess physical sensor telemetry hygiene (missing values, dropouts, extreme spikes).
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        issues: List[str] = []

        total_values = current_data.size
        nan_count = int(current_data.isna().sum().sum())
        inf_count = int(np.isinf(current_data.select_dtypes(include=[np.number])).sum().sum())

        if nan_count > 0:
            issues.append(f"Detected {nan_count} missing/NaN values ({nan_count/total_values:.2%})")
        if inf_count > 0:
            issues.append(f"Detected {inf_count} infinite numeric values")

        # Sensor dropouts check (zeros in vibration RMS or current)
        dropout_ratio = 0.0
        if "rad_rms" in current_data.columns:
            dropouts = int((current_data["rad_rms"] == 0.0).sum())
            dropout_ratio = dropouts / len(current_data) if len(current_data) > 0 else 0.0
            if dropout_ratio > 0.05:
                issues.append(f"Vibration sensor dropout ratio: {dropout_ratio:.1%} (> 5% threshold)")

        # Sensor spike check (> 15g threshold for Class 6: Sensor Anomaly)
        spike_count = 0
        if "rad_peak" in current_data.columns:
            spike_count = int((current_data["rad_peak"] > ClassThresholds.SENSOR_SPIKE_THRESHOLD_G).sum())
            if spike_count > 0:
                issues.append(f"Detected {spike_count} extreme vibration spikes exceeding {ClassThresholds.SENSOR_SPIKE_THRESHOLD_G}g")

        has_critical = len(issues) > 0
        return DataQualityResult(
            timestamp=timestamp,
            has_critical_issues=has_critical,
            nan_count=nan_count,
            infinite_count=inf_count,
            dropout_ratio=dropout_ratio,
            spike_count=spike_count,
            issues=issues,
        )

    def generate_html_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None,
        title: str = "Industrial IoT Data Drift Audit",
    ) -> str:
        """
        Generate standalone interactive HTML audit dashboard and save to disk.
        """
        if self.reference_data is None:
            raise RuntimeError("Reference data not initialized. Call set_reference_data() first.")

        ref_clean = self._select_numerical_features(self.reference_data)
        curr_clean = self._select_numerical_features(current_data)
        common_cols = [c for c in ref_clean.columns if c in curr_clean.columns]
        ref_clean = ref_clean[common_cols]
        curr_clean = curr_clean[common_cols]

        report = Report(metrics=[
            DataDriftPreset(drift_share=self.drift_share_threshold),
            DataQualityPreset(),
        ])
        report.run(reference_data=ref_clean, current_data=curr_clean)

        if output_path is None:
            filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            output_path = os.path.join(self.reports_dir, filename)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        report.save_html(output_path)
        return os.path.abspath(output_path)

    def generate_json_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete structured JSON report.
        """
        if self.reference_data is None:
            raise RuntimeError("Reference data not initialized. Call set_reference_data() first.")

        ref_clean = self._select_numerical_features(self.reference_data)
        curr_clean = self._select_numerical_features(current_data)
        common_cols = [c for c in ref_clean.columns if c in curr_clean.columns]

        report = Report(metrics=[DataDriftPreset(drift_share=self.drift_share_threshold)])
        report.run(reference_data=ref_clean[common_cols], current_data=curr_clean[common_cols])
        result = report.as_dict()

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        return result

    def check_retraining_trigger(
        self,
        drift_result: DriftResult,
        quality_result: Optional[DataQualityResult] = None,
    ) -> RetrainingDecision:
        """
        Evaluate if data drift or sensor degradation justifies triggering closed-loop automated retraining.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # If data quality has severe sensor spikes or dropouts (Sensor Anomaly - Class 6)
        if quality_result and quality_result.has_critical_issues:
            if quality_result.spike_count > 0 or quality_result.dropout_ratio > 0.10:
                return RetrainingDecision(
                    trigger_retraining=False,
                    severity="CRITICAL",
                    reason=(
                        "Sensor Hardware Anomaly detected (Class 6: transducer dropouts or extreme spikes). "
                        "Retraining suppressed to prevent fitting on corrupted telemetry. Technician inspection required."
                    ),
                    timestamp=timestamp,
                    metrics_snapshot=quality_result.to_dict(),
                )

        # Standard physical feature drift check
        if drift_result.dataset_drift:
            severity = "HIGH" if drift_result.drift_share >= 0.50 else "MEDIUM"
            drifted_str = ", ".join(drift_result.drifted_features[:6])
            if len(drift_result.drifted_features) > 6:
                drifted_str += f" and {len(drift_result.drifted_features) - 6} more"

            reason = (
                f"Statistically significant data drift detected in {drift_result.number_of_drifted_features}/"
                f"{drift_result.total_features} features (drift share: {drift_result.drift_share:.1%}, "
                f"threshold: {self.drift_share_threshold:.1%}). Drifted features: [{drifted_str}]. "
                "Triggering automated model retraining pipeline."
            )

            return RetrainingDecision(
                trigger_retraining=True,
                severity=severity,
                reason=reason,
                timestamp=timestamp,
                metrics_snapshot=drift_result.to_dict(),
            )

        return RetrainingDecision(
            trigger_retraining=False,
            severity="NONE",
            reason="Telemetry distributions aligned within acceptable statistical thresholds. Model performance optimal.",
            timestamp=timestamp,
            metrics_snapshot=drift_result.to_dict(),
        )
