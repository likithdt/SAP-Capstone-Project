"""
Unified MLOps & Drift Monitoring Pipeline.
Integrates MLflow experiment tracking, model registry promotion, Evidently AI drift monitoring,
and closed-loop automated retraining workflows for Industrial IoT prognostics.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timezone
import os
import numpy as np
import pandas as pd
import mlflow

from src.config import MLOpsConfig, DEFAULT_MLOPS_CONFIG, AssetConfig, DEFAULT_ASSET_CONFIG
from src.monitoring.mlflow_tracker import MLflowTracker
from src.monitoring.drift_detector import (
    EvidentlyDriftDetector,
    DriftResult,
    DataQualityResult,
    RetrainingDecision,
)
from src.models.engine import PrognosticsEngine
from src.models.isolation_forest import IsolationForestBaseline
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.models.cnn_classifier import FaultClassifier
from src.models.cnn_lstm_regressor import RULPredictor


class MLOpsPipeline:
    """
    End-to-End MLOps Controller.
    Orchestrates MLflow tracking, model lifecycle management, and Evidently AI drift monitoring.
    """

    def __init__(
        self,
        config: Optional[MLOpsConfig] = None,
        asset_config: Optional[AssetConfig] = None,
        tracker: Optional[MLflowTracker] = None,
        drift_detector: Optional[EvidentlyDriftDetector] = None,
        engine: Optional[PrognosticsEngine] = None,
    ):
        self.config = config or DEFAULT_MLOPS_CONFIG
        self.asset_config = asset_config or DEFAULT_ASSET_CONFIG
        self.tracker = tracker or MLflowTracker(config=self.config)
        self.drift_detector = drift_detector or EvidentlyDriftDetector(config=self.config)
        self.engine = engine or PrognosticsEngine(config=self.asset_config)

    def train_and_track_engine(
        self,
        features_df: pd.DataFrame,
        sequences_x: np.ndarray,
        classes_y: np.ndarray,
        rul_targets_y: Optional[np.ndarray] = None,
        run_name: str = "prognostics_full_training_run",
        register_model: bool = True,
        promote_to_stage: Optional[str] = "Production",
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> Tuple[PrognosticsEngine, str]:
        """
        Train all prognostics models with complete MLflow tracking, parameter/metric logging,
        artifact persistence, and model registry registration.
        """
        # Ensure sequences format is (N, channels, seq_len)
        if sequences_x.ndim == 3 and sequences_x.shape[1] > sequences_x.shape[2]:
            sequences_x = np.transpose(sequences_x, (0, 2, 1))

        # Synthetic RUL targets if not supplied
        if rul_targets_y is None:
            # Healthy: ~720 hrs; Degrading: lower hours depending on class
            base_rul = np.array([720.0, 48.0, 120.0, 160.0, 72.0, 96.0, 360.0])
            rul_targets_y = np.array([base_rul[int(c)] for c in classes_y], dtype=np.float32)

        healthy_mask = (classes_y == 0)
        healthy_features = features_df[healthy_mask] if np.any(healthy_mask) else features_df
        healthy_seqs_x = sequences_x[healthy_mask] if np.any(healthy_mask) else sequences_x

        # Temporal shape for Autoencoder: (N, seq_len, channels)
        healthy_temporal = np.transpose(healthy_seqs_x, (0, 2, 1))

        with self.tracker.track_run(run_name=run_name, tags={"framework": "PyTorch + Scikit-learn", "pipeline": "Prognostics"}) as run:
            run_id = run.info.run_id

            # 1. Log Asset & Training Parameters
            self.tracker.log_params({
                "asset_id": self.asset_config.asset_id,
                "nominal_speed_rpm": self.asset_config.nominal_speed_rpm,
                "sampling_rate_hz": self.asset_config.sampling_rate_hz,
                "training_samples": len(features_df),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
            })

            # 2. Train Isolation Forest
            self.engine.iso_forest.fit(healthy_features)
            iso_anom, iso_scores = self.engine.iso_forest.predict_anomaly(healthy_features)
            iso_anom_rate = float(np.mean(iso_anom))
            self.tracker.log_isolation_forest(self.engine.iso_forest)
            self.tracker.log_metric("iso_forest_baseline_anomaly_rate", iso_anom_rate)

            # 3. Train LSTM Autoencoder
            self.engine.autoencoder.fit(
                healthy_temporal,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                verbose=False,
            )
            ae_anoms, ae_scores = self.engine.autoencoder.predict_anomaly(healthy_temporal)
            self.tracker.log_metrics({
                "lstm_ae_threshold": float(self.engine.autoencoder.reconstruction_threshold),
                "lstm_ae_baseline_mse_mean": float(np.mean(ae_scores)),
                "lstm_ae_baseline_anomaly_rate": float(np.mean(ae_anoms)),
            })

            # 4. Train 1D-CNN Fault Classifier
            self.engine.classifier.fit(
                sequences_x,
                classes_y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                verbose=False,
            )
            pred_classes, confidences, _ = self.engine.classifier.predict(sequences_x)
            classifier_accuracy = float(np.mean(pred_classes == classes_y))
            self.tracker.log_metrics({
                "cnn_classifier_train_accuracy": classifier_accuracy,
                "cnn_classifier_mean_confidence": float(np.mean(confidences)),
            })

            # 5. Train CNN-LSTM RUL Regressor
            self.engine.rul_predictor.fit(
                sequences_x,
                rul_targets_y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                verbose=False,
            )
            pred_ruls = self.engine.rul_predictor.predict(sequences_x)
            rul_mae = float(np.mean(np.abs(pred_ruls - rul_targets_y)))
            rul_rmse = float(np.sqrt(np.mean((pred_ruls - rul_targets_y) ** 2)))
            self.tracker.log_metrics({
                "rul_train_mae_hours": rul_mae,
                "rul_train_rmse_hours": rul_rmse,
            })

            # 6. Log Complete Prognostics Engine Bundle
            metadata = {
                "asset_id": self.asset_config.asset_id,
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "classifier_accuracy": classifier_accuracy,
                "rul_mae_hours": rul_mae,
                "run_id": run_id,
            }
            self.tracker.log_prognostics_engine(self.engine, metadata=metadata)

            # 7. Register and Promote Model Version
            if register_model:
                try:
                    reg_model = self.tracker.register_model(
                        run_id=run_id,
                        model_name=self.config.model_registry_name,
                        artifact_path="prognostics_engine",
                    )
                    version = reg_model.version
                    if promote_to_stage:
                        self.tracker.transition_model_stage(
                            version=version,
                            stage=promote_to_stage,
                            model_name=self.config.model_registry_name,
                        )
                except Exception:
                    # Model registry fallback if standalone tracking store
                    pass

        # 8. Set baseline reference dataset for Evidently AI drift monitoring
        self.establish_reference_baseline(healthy_features)

        return self.engine, run_id

    def establish_reference_baseline(self, reference_features_df: pd.DataFrame) -> None:
        """Initialize or update reference baseline distribution for drift monitoring."""
        self.drift_detector.set_reference_data(reference_features_df)

    def monitor_telemetry_batch(
        self,
        features_df: pd.DataFrame,
        step: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate live telemetry features for data drift, quality issues, and retraining trigger.
        Optionally logs metrics to active MLflow run.
        """
        drift_res = self.drift_detector.evaluate_data_drift(features_df)
        quality_res = self.drift_detector.evaluate_data_quality(features_df)
        retrain_decision = self.drift_detector.check_retraining_trigger(drift_res, quality_res)

        # Log drift metrics to MLflow if tracking
        mlflow_logged = False
        try:
            drift_metrics = {
                "evidently_drift_share": drift_res.drift_share,
                "evidently_drifted_features_count": float(drift_res.number_of_drifted_features),
                "evidently_dataset_drift_detected": 1.0 if drift_res.dataset_drift else 0.0,
                "evidently_retraining_triggered": 1.0 if retrain_decision.trigger_retraining else 0.0,
                "telemetry_nan_count": float(quality_res.nan_count),
                "telemetry_dropout_ratio": quality_res.dropout_ratio,
                "telemetry_spike_count": float(quality_res.spike_count),
            }
            if mlflow.active_run():
                self.tracker.log_metrics(drift_metrics, step=step)
                mlflow_logged = True
            elif run_id:
                # Log to specified run using MlflowClient
                for k, v in drift_metrics.items():
                    self.tracker.client.log_metric(run_id=run_id, key=k, value=v, step=step)
                mlflow_logged = True
        except Exception:
            mlflow_logged = False

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_result": drift_res.to_dict(),
            "data_quality": quality_res.to_dict(),
            "retraining_decision": retrain_decision.to_dict(),
            "mlflow_logged": mlflow_logged,
        }

    def trigger_retraining_workflow(
        self,
        trigger_reason: str,
        features_df: pd.DataFrame,
        sequences_x: np.ndarray,
        classes_y: np.ndarray,
        rul_targets_y: Optional[np.ndarray] = None,
        epochs: int = 10,
    ) -> Tuple[PrognosticsEngine, str]:
        """
        Autonomous closed-loop retraining run invoked when drift thresholds are breached.
        """
        run_name = f"closed_loop_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {
            "retraining_trigger": "True",
            "trigger_reason": trigger_reason,
            "automated_closed_loop": "True",
        }

        # Execute training with tags
        engine, run_id = self.train_and_track_engine(
            features_df=features_df,
            sequences_x=sequences_x,
            classes_y=classes_y,
            rul_targets_y=rul_targets_y,
            run_name=run_name,
            register_model=True,
            promote_to_stage="Production",
            epochs=epochs,
        )

        return engine, run_id
