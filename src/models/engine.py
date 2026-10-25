"""
Unified Prognostics Inference Engine.
Combines Isolation Forest, LSTM Autoencoder, 1D-CNN Fault Classifier, and CNN-LSTM RUL Regressor
into a standalone inference engine producing standardized diagnostic predictions matching SAP Capstone specifications.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import numpy as np
import pandas as pd

from src.config import AssetConfig, ClassThresholds, FaultClass, DEFAULT_ASSET_CONFIG
from src.features.pipeline import FeatureExtractionPipeline
from src.models.isolation_forest import IsolationForestBaseline
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.models.cnn_classifier import FaultClassifier
from src.models.cnn_lstm_regressor import RULPredictor


@dataclass
class DiagnosticReport:
    """Standardized prognostic diagnostic report matching Digital Twin and SAP schemas."""
    asset_id: str
    timestamp: str
    is_anomaly: bool
    anomaly_score: float             # Normalized composite anomaly score in [0.0, 1.0]
    risk_level: str                  # LOW, MEDIUM, HIGH, CRITICAL
    predicted_class: int             # Fault class 0-6
    predicted_failure_mode: str      # Human-readable failure mode
    confidence: float                # Multi-class classifier confidence
    rul_hours: float                 # Predicted Remaining Useful Life in operating hours
    reconstruction_error: float      # LSTM Autoencoder MSE error
    isolation_forest_score: float    # Isolation Forest anomaly score
    evidence: List[str]              # Diagnostic root-cause evidence bullets for RAG/Agentic Swarm
    recommended_action: str          # Recommended maintenance action for SAP PM
    features: Dict[str, Any] = field(default_factory=dict)
    class_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "asset_id": self.asset_id,
            "timestamp": self.timestamp,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "risk_level": self.risk_level,
            "predicted_class": self.predicted_class,
            "predicted_failure_mode": self.predicted_failure_mode,
            "confidence": round(self.confidence, 4),
            "rul_hours": round(self.rul_hours, 1),
            "reconstruction_error": round(self.reconstruction_error, 5),
            "isolation_forest_score": round(self.isolation_forest_score, 4),
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "class_probabilities": self.class_probabilities,
        }


class PrognosticsEngine:
    """
    Standalone Multimodal Prognostics Inference Engine.
    Executes feature engineering and evaluates:
      - Isolation Forest (Tabular baseline)
      - LSTM Autoencoder (Temporal reconstruction anomaly detector)
      - 1D-CNN (Supervised 7-class fault classifier)
      - CNN-LSTM (Degradation sequence RUL regressor)
    """

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        feature_pipeline: Optional[FeatureExtractionPipeline] = None,
        iso_forest: Optional[IsolationForestBaseline] = None,
        autoencoder: Optional[LSTMAutoencoderDetector] = None,
        classifier: Optional[FaultClassifier] = None,
        rul_predictor: Optional[RULPredictor] = None,
    ):
        self.config = config or DEFAULT_ASSET_CONFIG
        self.feature_pipeline = feature_pipeline or FeatureExtractionPipeline(self.config)
        self.iso_forest = iso_forest or IsolationForestBaseline()
        self.autoencoder = autoencoder or LSTMAutoencoderDetector(in_channels=3)
        self.classifier = classifier or FaultClassifier(in_channels=3, num_classes=7)
        self.rul_predictor = rul_predictor or RULPredictor(in_channels=3)

    def save(self, save_dir: str) -> None:
        """Serialize all engine model components to directory."""
        os.makedirs(save_dir, exist_ok=True)
        self.iso_forest.save(os.path.join(save_dir, "iso_forest.joblib"))
        self.autoencoder.save(os.path.join(save_dir, "lstm_autoencoder.pt"))
        self.classifier.save(os.path.join(save_dir, "cnn_classifier.pt"))
        self.rul_predictor.save(os.path.join(save_dir, "rul_predictor.pt"))

    @classmethod
    def load(cls, save_dir: str, config: Optional[AssetConfig] = None) -> "PrognosticsEngine":
        """Load all engine model components from directory."""
        iso_forest = IsolationForestBaseline.load(os.path.join(save_dir, "iso_forest.joblib"))
        autoencoder = LSTMAutoencoderDetector.load(os.path.join(save_dir, "lstm_autoencoder.pt"))
        classifier = FaultClassifier.load(os.path.join(save_dir, "cnn_classifier.pt"))
        rul_predictor = RULPredictor.load(os.path.join(save_dir, "rul_predictor.pt"))
        return cls(
            config=config,
            iso_forest=iso_forest,
            autoencoder=autoencoder,
            classifier=classifier,
            rul_predictor=rul_predictor,
        )

    def predict(
        self,
        accel_x: np.ndarray,
        accel_y: Optional[np.ndarray] = None,
        accel_z: Optional[np.ndarray] = None,
        temperature_c: Optional[np.ndarray] = None,
        current_a: Optional[np.ndarray] = None,
        speed_rpm: Optional[np.ndarray] = None,
    ) -> DiagnosticReport:
        """
        Execute full prognostics pipeline on raw sensor telemetry window.
        """
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Extract tabular features
        features = self.feature_pipeline.extract_from_arrays(
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            temperature_c=temperature_c,
            current_a=current_a,
            speed_rpm=speed_rpm,
        )

        # 2. Format 3-channel vibration array for deep learning models: (channels, seq_len)
        n = len(accel_x)
        ay = accel_y if accel_y is not None else np.zeros(n)
        az = accel_z if accel_z is not None else np.zeros(n)
        clean_x = np.nan_to_num(accel_x, nan=0.0)
        clean_y = np.nan_to_num(ay, nan=0.0)
        clean_z = np.nan_to_num(az, nan=0.0)

        # Shape (channels, seq_len): (3, n)
        vibration_channels = np.stack([clean_x, clean_y, clean_z], axis=0)
        # Shape (seq_len, channels): (n, 3) for LSTM Autoencoder
        vibration_temporal = np.stack([clean_x, clean_y, clean_z], axis=1)

        # 3. Model 1: Isolation Forest
        if self.iso_forest.is_fitted:
            iso_res = self.iso_forest.score_single(features)
            iso_score = iso_res["anomaly_score"]
            iso_anom = iso_res["is_anomaly"]
        else:
            # Rule-based fallback if not fitted
            iso_score = 0.85 if features["rad_rms"] > 0.40 or features["temp_max"] > 75.0 else 0.15
            iso_anom = iso_score > 0.5

        # 4. Model 2: LSTM Autoencoder
        if self.autoencoder.is_fitted:
            ae_res = self.autoencoder.score_single_window(vibration_temporal)
            recon_err = ae_res["reconstruction_error"]
            ae_score = ae_res["anomaly_score"]
            ae_anom = ae_res["is_anomaly"]
        else:
            recon_err = 0.045 if features["rad_rms"] > 0.40 else 0.008
            ae_score = min(1.0, recon_err / 0.035)
            ae_anom = ae_score > 0.5

        # 5. Model 3: 1D-CNN Fault Classifier
        if self.classifier.is_fitted:
            cnn_res = self.classifier.predict_single(vibration_channels)
            pred_class = cnn_res["predicted_class"]
            pred_label = cnn_res["predicted_label"]
            confidence = cnn_res["confidence"]
            class_probs = cnn_res["class_probabilities"]
        else:
            # Physics-based classification fallback if unweighted
            pred_class, pred_label, confidence, class_probs = self._rule_based_classifier(features)

        # 6. Model 4: CNN-LSTM RUL Regressor
        if self.rul_predictor.is_fitted:
            rul_hours = self.rul_predictor.predict_single(vibration_channels)
        else:
            rul_hours = self._estimate_rul_heuristic(pred_class, features)

        # 7. Synthesize Composite Anomaly Score & Multi-modal Risk Fusion
        # Matches Section 9.3: HealthScore = 100 - (w1*A + w2*T + w3*C + w4*R)
        vib_score = float(np.clip(0.60 * ae_score + 0.40 * iso_score, 0.0, 1.0))
        
        # Thermal risk: 0.0 at healthy <= 52 °C, climbing to 1.0 at critical 85 °C
        temp_max = features.get("temp_max", 50.0)
        thermal_risk = float(np.clip((temp_max - 52.0) / (ClassThresholds.THERMAL_OVERHEATING_CRITICAL_C - 52.0), 0.0, 1.0))
        
        # Current risk: 0.0 at nominal <= 1.30 A, climbing to 1.0 at overload 2.4 A
        curr_mean = features.get("current_mean", 1.20)
        current_risk = float(np.clip((curr_mean - 1.30) / (ClassThresholds.OVERLOAD_CURRENT_MIN_A - 1.30), 0.0, 1.0))

        # Sensor glitch risk
        glitch_risk = 1.0 if features.get("integrity_has_sensor_glitch", 0) == 1 else 0.0

        # Unified composite anomaly score taking worst-case severity across modalities
        composite_anomaly_score = float(np.clip(max(vib_score, thermal_risk * 0.92, current_risk * 0.92, glitch_risk), 0.0, 1.0))
        is_anomaly = bool(composite_anomaly_score > 0.40 or pred_class != 0 or features["integrity_has_sensor_glitch"] == 1)

        if composite_anomaly_score < 0.35:
            risk_level = "LOW"
        elif composite_anomaly_score < 0.65:
            risk_level = "MEDIUM"
        elif composite_anomaly_score < 0.85:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # 8. Generate Contextual Diagnostic Evidence & Recommended Action
        evidence = self._generate_diagnostic_evidence(features, pred_class, composite_anomaly_score)
        action = self._determine_recommended_action(pred_class, risk_level)

        return DiagnosticReport(
            asset_id=self.config.asset_id,
            timestamp=now_ts,
            is_anomaly=is_anomaly,
            anomaly_score=composite_anomaly_score,
            risk_level=risk_level,
            predicted_class=pred_class,
            predicted_failure_mode=pred_label,
            confidence=confidence,
            rul_hours=rul_hours,
            reconstruction_error=recon_err,
            isolation_forest_score=iso_score,
            evidence=evidence,
            recommended_action=action,
            features=features,
            class_probabilities=class_probs,
        )

    def predict_batch(self, batch: Any) -> DiagnosticReport:
        """Convenience method to score a TelemetryBatch instance."""
        return self.predict(
            accel_x=batch.accel_x,
            accel_y=batch.accel_y,
            accel_z=batch.accel_z,
            temperature_c=batch.temperature_c,
            current_a=batch.current_a,
            speed_rpm=batch.speed_rpm,
        )

    def _rule_based_classifier(self, feats: Dict[str, Any]) -> Tuple[int, str, float, Dict[str, float]]:
        """Physical rule-based diagnostic classifier matching Capstone physics."""
        probs = {FaultClass(c).label: 0.05 for c in range(7)}

        if feats.get("integrity_has_sensor_glitch", 0) == 1:
            cls = FaultClass.SENSOR_ANOMALY
            conf = 0.95
        elif feats.get("temp_critical_flag", 0) == 1 or feats.get("temp_max", 0) > 80.0:
            cls = FaultClass.THERMAL_OVERHEATING
            conf = 0.92
        elif feats.get("current_overload_flag", 0) == 1 or feats.get("current_mean", 0) > 2.2:
            cls = FaultClass.MOTOR_OVERLOAD
            conf = 0.89
        elif feats.get("rad_kurtosis", 0) > 4.5 and feats.get("rad_crest_factor", 0) >= 3.5:
            cls = FaultClass.BEARING_WEAR
            conf = 0.88
        elif feats.get("axial_to_radial_rms_ratio", 0) > 0.55 and feats.get("rad_band_2x_energy", 0) > 0.4 * feats.get("rad_band_1x_energy", 1):
            cls = FaultClass.SHAFT_MISALIGNMENT
            conf = 0.86
        elif feats.get("rad_rms", 0) > 0.70 and feats.get("rad_dominant_freq_hz", 0) == 30.0:
            cls = FaultClass.ROTOR_IMBALANCE
            conf = 0.90
        else:
            cls = FaultClass.HEALTHY
            conf = 0.94

        probs[cls.label] = conf
        # Normalize
        total = sum(probs.values())
        probs = {k: float(np.round(v / total, 4)) for k, v in probs.items()}
        return int(cls), cls.label, conf, probs

    def _estimate_rul_heuristic(self, pred_class: int, feats: Dict[str, Any]) -> float:
        """Physical heuristic RUL projection based on failure criticality and degradation pace."""
        if pred_class == FaultClass.HEALTHY:
            return 120.0
        elif pred_class == FaultClass.BEARING_WEAR:
            # Accelerated spalling: 24 to 48 hours typical remaining
            kurt = feats.get("rad_kurtosis", 5.0)
            return float(np.clip(60.0 - 2.5 * kurt, 12.0, 48.0))
        elif pred_class == FaultClass.ROTOR_IMBALANCE:
            return 72.0
        elif pred_class == FaultClass.SHAFT_MISALIGNMENT:
            return 64.0
        elif pred_class == FaultClass.THERMAL_OVERHEATING:
            temp = feats.get("temp_max", 85.0)
            return float(np.clip(40.0 - (temp - 80.0) * 2.0, 4.0, 24.0))
        elif pred_class == FaultClass.MOTOR_OVERLOAD:
            return 8.0
        elif pred_class == FaultClass.SENSOR_ANOMALY:
            return 0.0
        return 50.0

    def _generate_diagnostic_evidence(
        self, feats: Dict[str, Any], pred_class: int, anomaly_score: float
    ) -> List[str]:
        """Generate structured diagnostic root-cause evidence for RAG Diagnostic Agent."""
        evidence = []
        rms = feats.get("rad_rms", 0.0)
        kurt = feats.get("rad_kurtosis", 0.0)
        dom_freq = feats.get("rad_dominant_freq_hz", 0.0)
        temp_mean = feats.get("temp_mean", 0.0)
        curr_mean = feats.get("current_mean", 0.0)

        if rms > 0.35:
            delta_pct = int(((rms - 0.25) / 0.25) * 100)
            evidence.append(f"RMS vibration elevated to {rms:.3f} g (+{delta_pct}% above healthy 0.25 g baseline)")
        else:
            evidence.append(f"RMS vibration nominal at {rms:.3f} g (healthy range 0.15–0.35 g)")

        if kurt > 4.5:
            evidence.append(f"Kurtosis elevated to {kurt:.2f} (> 4.5 threshold), signaling periodic impact impulses")

        if pred_class == FaultClass.BEARING_WEAR and feats.get("rad_envelope_bpfo_ratio", 0) > 1.8:
            evidence.append("Prominent envelope spectrum peak at BPFO defect frequency (107.4 Hz)")

        if pred_class == FaultClass.ROTOR_IMBALANCE:
            evidence.append(f"Dominant vibration concentrated strictly at 1X rotational frequency ({dom_freq:.1f} Hz)")

        if pred_class == FaultClass.SHAFT_MISALIGNMENT:
            ax_ratio = feats.get("axial_to_radial_rms_ratio", 0.0)
            evidence.append(f"High axial-to-radial vibration ratio ({ax_ratio:.2f}) and pronounced 2X harmonic")

        if feats.get("temp_critical_flag", 0) == 1 or temp_mean > 70.0:
            evidence.append(f"Severe thermal drift detected: maximum temperature reached {feats.get('temp_max', 0):.1f} °C (> 85 °C critical)")

        if feats.get("current_overload_flag", 0) == 1:
            evidence.append(f"Abnormal motor current draw of {curr_mean:.2f} A with 100 Hz supply ripple")

        if feats.get("integrity_has_sensor_glitch", 0) == 1:
            evidence.append(f"Sensor anomaly detected: {feats.get('integrity_extreme_spike_count', 0)} spikes > 15g or signal dropouts")

        return evidence

    def _determine_recommended_action(self, pred_class: int, risk_level: str) -> str:
        """Map failure mode to standardized SAP PM/MM operational action."""
        actions = {
            FaultClass.HEALTHY: "Normal operation. Continuous IoT telemetry monitoring active.",
            FaultClass.BEARING_WEAR: "Inspect bearing and schedule replacement (SAP PM Work Order - 6205-2RS Bearing).",
            FaultClass.ROTOR_IMBALANCE: "Rebalance rotor assembly and check dynamic balancing weights.",
            FaultClass.SHAFT_MISALIGNMENT: "Perform laser alignment on motor and driven shaft coupling.",
            FaultClass.THERMAL_OVERHEATING: "Inspect cooling fan airflow, winding resistance, and bearing lubrication.",
            FaultClass.MOTOR_OVERLOAD: "Reduce mechanical conveyor load, verify gearbox clearances, and inspect motor drive circuit.",
            FaultClass.SENSOR_ANOMALY: "Inspect accelerometer physical mounting, wiring harness, and ADC connection.",
        }
        return actions.get(FaultClass(pred_class), "Inspect asset condition.")


if __name__ == "__main__":
    import json
    from src.digital_twin.simulator import PhysicalSignalSimulator

    print("=" * 80)
    print("PROGNOSTICS ENGINE: REAL-TIME INFERENCE DEMONSTRATION")
    print("=" * 80)
    sim = PhysicalSignalSimulator()
    engine = PrognosticsEngine()

    for fc in FaultClass:
        batch = sim.generate_window(fc, duration_sec=1.0, seed=42, allow_nan=(fc == FaultClass.SENSOR_ANOMALY))
        report = engine.predict_batch(batch)
        print(f"\n[{fc.name}] -> Predicted: {report.predicted_failure_mode} (Confidence: {report.confidence*100:.1f}%)")
        print(f"  Anomaly Score: {report.anomaly_score:.3f} | Risk Level: {report.risk_level} | Is Anomaly: {report.is_anomaly}")
        print(f"  Projected RUL: {report.rul_hours:.1f} hours | Reconstruction Error: {report.reconstruction_error:.5f}")
        print(f"  Evidence: {report.evidence[:2]}")
        print(f"  Recommended SAP Action: {report.recommended_action}")
    print("\n" + "=" * 80)

