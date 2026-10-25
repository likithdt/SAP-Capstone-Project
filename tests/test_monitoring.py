"""
Unit and Integration Tests for MLOps Tracking and Drift Monitoring.
Covers MLflow Experiment Tracking, Model Registry, Evidently AI Data Drift, Target Drift,
Data Quality Assessment, and Automated Closed-Loop Retraining Workflows.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.config import AssetConfig, FaultClass, MLOpsConfig
from src.models.engine import PrognosticsEngine
from src.models.isolation_forest import IsolationForestBaseline
from src.monitoring.mlflow_tracker import MLflowTracker
from src.monitoring.drift_detector import (
    EvidentlyDriftDetector,
    DriftResult,
    DataQualityResult,
    RetrainingDecision,
)
from src.monitoring.pipeline import MLOpsPipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts and SQLite DBs."""
    d = tempfile.mkdtemp(prefix="test_mlops_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_features_df():
    """Load or generate a standard 7-class feature DataFrame for tests."""
    csv_path = "data/processed/engineered_features.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback synthetic features
        n_per_class = 20
        rows = []
        for fc in range(7):
            for _ in range(n_per_class):
                rows.append({
                    "fault_class": fc,
                    "rad_rms": 0.25 if fc == 0 else (0.85 if fc in [1, 2, 5] else 0.35),
                    "rad_peak": 0.50 if fc == 0 else (2.5 if fc in [1, 2] else 0.8),
                    "rad_crest_factor": 2.5 if fc == 0 else (5.5 if fc == 1 else 2.8),
                    "rad_kurtosis": 3.0 if fc == 0 else (6.2 if fc == 1 else 3.1),
                    "rad_dominant_freq_hz": 30.0,
                    "rad_spectral_centroid": 120.0,
                    "rad_spectral_spread": 45.0,
                    "rad_spectral_entropy": 0.85,
                    "rad_band_1x_energy": 0.05 if fc != 2 else 0.45,
                    "rad_band_2x_energy": 0.02 if fc != 3 else 0.30,
                    "rad_band_3x_energy": 0.01,
                    "rad_band_bpfo_energy": 0.01 if fc != 1 else 0.35,
                    "axial_to_radial_rms_ratio": 0.3 if fc != 3 else 0.75,
                    "temp_mean": 48.0 if fc != 4 else 78.0,
                    "temp_slope": 0.01 if fc != 4 else 0.25,
                    "current_mean": 1.2 if fc != 5 else 2.8,
                    "current_100hz_ripple": 0.05 if fc != 5 else 0.35,
                })
        df = pd.DataFrame(rows)
    return df


# =============================================================================
# 1. MLflowTracker Tests
# =============================================================================

def test_mlflow_tracker_initialization(temp_dir):
    """Verify MLflow tracker sets up SQLite tracking store and experiment."""
    db_file = os.path.join(temp_dir, "test_mlflow.db").replace("\\", "/")
    uri = f"sqlite:///{db_file}"
    exp_name = "Test-IoT-Prognostics"

    tracker = MLflowTracker(tracking_uri=uri, experiment_name=exp_name)
    assert tracker.experiment_id is not None
    assert tracker.client.get_experiment_by_name(exp_name) is not None


def test_mlflow_tracker_run_and_logging(temp_dir):
    """Verify context manager run creation and parameter/metric/artifact logging."""
    db_file = os.path.join(temp_dir, "test_mlflow.db").replace("\\", "/")
    uri = f"sqlite:///{db_file}"
    tracker = MLflowTracker(tracking_uri=uri, experiment_name="Test-Run-Logging")

    with tracker.track_run(run_name="unit_test_run", tags={"environment": "pytest"}) as run:
        run_id = run.info.run_id
        tracker.log_param("model_type", "1D-CNN")
        tracker.log_params({"batch_size": 32, "lr": 0.001})
        tracker.log_metric("accuracy", 0.965)
        tracker.log_metrics({"f1_macro": 0.952, "loss": 0.124})

        sample_dict = {"accuracy_per_class": [0.98, 0.95, 0.97]}
        tracker.log_dict(sample_dict, "class_metrics.json")

    client_run = tracker.client.get_run(run_id)
    assert client_run.info.status == "FINISHED"
    assert client_run.data.params["model_type"] == "1D-CNN"
    assert client_run.data.params["batch_size"] == "32"
    assert pytest.approx(client_run.data.metrics["accuracy"], 0.001) == 0.965
    assert pytest.approx(client_run.data.metrics["f1_macro"], 0.001) == 0.952


def test_mlflow_tracker_model_and_engine_logging(temp_dir, sample_features_df):
    """Verify logging Isolation Forest and complete PrognosticsEngine bundle."""
    db_file = os.path.join(temp_dir, "test_mlflow.db").replace("\\", "/")
    uri = f"sqlite:///{db_file}"
    tracker = MLflowTracker(tracking_uri=uri, experiment_name="Test-Model-Logging")

    healthy_df = sample_features_df[sample_features_df["fault_class"] == 0]

    with tracker.track_run(run_name="model_artifacts_test") as run:
        # Fit and log Isolation Forest
        iso = IsolationForestBaseline().fit(healthy_df)
        tracker.log_isolation_forest(iso)

        # Log complete engine bundle
        engine = PrognosticsEngine()
        tracker.log_prognostics_engine(engine, metadata={"model_bundle_version": "1.0"})

        run_id = run.info.run_id

    # Verify run completed
    run_info = tracker.client.get_run(run_id)
    assert run_info.info.status == "FINISHED"


# =============================================================================
# 2. EvidentlyDriftDetector Tests
# =============================================================================

def test_evidently_drift_detector_no_drift_on_healthy(sample_features_df, temp_dir):
    """Baseline healthy telemetry evaluated against similar healthy telemetry shows no drift."""
    healthy_df = sample_features_df[sample_features_df["fault_class"] == 0].copy()
    assert len(healthy_df) >= 20

    # Split healthy into reference (first half) and current (second half)
    half = len(healthy_df) // 2
    ref_df = healthy_df.iloc[:half]
    curr_df = healthy_df.iloc[half:]

    detector = EvidentlyDriftDetector(
        reference_data=ref_df,
        drift_share_threshold=0.30,
        reports_dir=temp_dir,
    )

    drift_res = detector.evaluate_data_drift(curr_df)
    assert isinstance(drift_res, DriftResult)
    assert not drift_res.dataset_drift
    assert drift_res.drift_share < 0.30

    retraining = detector.check_retraining_trigger(drift_res)
    assert isinstance(retraining, RetrainingDecision)
    assert not retraining.trigger_retraining
    assert retraining.severity == "NONE"


def test_evidently_drift_detector_detects_bearing_wear_drift(sample_features_df, temp_dir):
    """Evaluating bearing wear telemetry against healthy reference flags data drift & triggers retraining."""
    healthy_df = sample_features_df[sample_features_df["fault_class"] == 0]
    bearing_df = sample_features_df[sample_features_df["fault_class"] == 1]

    detector = EvidentlyDriftDetector(
        reference_data=healthy_df,
        drift_share_threshold=0.30,
        reports_dir=temp_dir,
    )

    drift_res = detector.evaluate_data_drift(bearing_df)
    assert drift_res.dataset_drift
    assert drift_res.drift_share >= 0.30
    assert len(drift_res.drifted_features) > 0

    # Key vibration features must have drifted
    drifted_set = set(drift_res.drifted_features)
    assert any(feat in drifted_set for feat in ["rad_rms", "rad_kurtosis", "rad_crest_factor", "rad_band_bpfo_energy"])

    # Verify closed-loop retraining decision
    retraining = detector.check_retraining_trigger(drift_res)
    assert retraining.trigger_retraining
    assert retraining.severity in ["HIGH", "MEDIUM"]
    assert "data drift detected" in retraining.reason.lower()


def test_evidently_drift_detector_detects_thermal_drift(sample_features_df, temp_dir):
    """Evaluating overheating telemetry against healthy reference flags thermal feature drift."""
    healthy_df = sample_features_df[sample_features_df["fault_class"] == 0]
    thermal_df = sample_features_df[sample_features_df["fault_class"] == 4]

    detector = EvidentlyDriftDetector(
        reference_data=healthy_df,
        drift_share_threshold=0.30,
        reports_dir=temp_dir,
    )

    drift_res = detector.evaluate_data_drift(thermal_df)
    assert drift_res.dataset_drift
    assert "temp_mean" in drift_res.drifted_features


def test_evidently_drift_detector_data_quality_and_sensor_anomaly(sample_features_df, temp_dir):
    """Sensor dropouts or extreme spikes are flagged by data quality checks and suppress unneeded retraining."""
    healthy_df = sample_features_df[sample_features_df["fault_class"] == 0].copy()

    # Corrupt data with sensor spikes (>15g) and NaN dropouts (Class 6: Sensor Anomaly)
    corrupted_df = healthy_df.copy()
    corrupted_df.loc[corrupted_df.index[:5], "rad_peak"] = 25.0  # >15g extreme spike
    corrupted_df.loc[corrupted_df.index[5:10], "rad_rms"] = 0.0  # dropouts

    detector = EvidentlyDriftDetector(reference_data=healthy_df, reports_dir=temp_dir)
    quality_res = detector.evaluate_data_quality(corrupted_df)

    assert isinstance(quality_res, DataQualityResult)
    assert quality_res.has_critical_issues
    assert quality_res.spike_count >= 5

    # Retraining trigger check with corrupted sensor data
    drift_res = detector.evaluate_data_drift(corrupted_df)
    retraining = detector.check_retraining_trigger(drift_res, quality_res)

    # Must NOT trigger model retraining for sensor glitches (Class 6: Sensor Anomaly)
    assert not retraining.trigger_retraining
    assert retraining.severity == "CRITICAL"
    assert "sensor hardware anomaly" in retraining.reason.lower()


def test_evidently_drift_detector_report_generation(sample_features_df, temp_dir):
    """Verify HTML dashboard export and structured JSON report generation."""
    healthy_df = sample_features_df[sample_features_df["fault_class"] == 0]
    bearing_df = sample_features_df[sample_features_df["fault_class"] == 1]

    detector = EvidentlyDriftDetector(reference_data=healthy_df, reports_dir=temp_dir)

    # 1. HTML Report
    html_path = os.path.join(temp_dir, "test_drift_report.html")
    generated_html = detector.generate_html_report(bearing_df, output_path=html_path)
    assert os.path.exists(generated_html)
    assert os.path.getsize(generated_html) > 1000

    # 2. JSON Report
    json_path = os.path.join(temp_dir, "test_drift_report.json")
    report_dict = detector.generate_json_report(bearing_df, output_path=json_path)
    assert os.path.exists(json_path)
    assert "metrics" in report_dict


def test_evidently_target_prediction_drift(sample_features_df):
    """Verify target / prediction drift detection."""
    detector = EvidentlyDriftDetector()

    ref_preds = pd.DataFrame({
        "anomaly_score": np.random.uniform(0.05, 0.20, 50),
        "rul_hours": np.random.uniform(650.0, 720.0, 50),
        "predicted_class": np.zeros(50),
    })

    curr_preds = pd.DataFrame({
        "anomaly_score": np.random.uniform(0.75, 0.95, 50),  # shifted!
        "rul_hours": np.random.uniform(20.0, 80.0, 50),      # shifted!
        "predicted_class": np.ones(50),
    })

    target_drift = detector.evaluate_target_drift(ref_preds, curr_preds)
    assert target_drift.dataset_drift
    assert "anomaly_score" in target_drift.drifted_features


# =============================================================================
# 3. MLOpsPipeline End-to-End Integration Tests
# =============================================================================

def test_mlops_pipeline_end_to_end(sample_features_df, temp_dir):
    """
    End-to-end integration test:
      1. Train prognostics models with MLflow tracking.
      2. Set reference baseline.
      3. Evaluate healthy telemetry batch (no drift).
      4. Evaluate fault telemetry batch (drift detected -> retraining triggered).
      5. Trigger closed-loop retraining workflow.
    """
    db_file = os.path.join(temp_dir, "test_pipeline_mlflow.db").replace("\\", "/")
    uri = f"sqlite:///{db_file}"
    mlops_cfg = MLOpsConfig(
        tracking_uri=uri,
        experiment_name="Test-MLOps-Pipeline",
        report_output_dir=os.path.join(temp_dir, "reports"),
    )

    pipeline = MLOpsPipeline(config=mlops_cfg)

    # Prepare small training dataset
    features = sample_features_df.copy()
    n_samples = len(features)
    classes_y = features["fault_class"].to_numpy(dtype=np.int64)

    # 3-channel synthetic vibration sequences (N, channels=3, seq_len=128)
    rng = np.random.default_rng(42)
    sequences_x = rng.normal(0.0, 0.2, size=(n_samples, 3, 128)).astype(np.float32)

    # Step 1: Train & Track Engine
    engine, run_id = pipeline.train_and_track_engine(
        features_df=features,
        sequences_x=sequences_x,
        classes_y=classes_y,
        run_name="integration_test_train",
        register_model=False,
        epochs=2,
        batch_size=16,
    )
    assert engine is not None
    assert run_id is not None

    # Step 2: Stream healthy batch -> verify no drift
    healthy_batch = features[features["fault_class"] == 0].iloc[:15]
    mon_res_healthy = pipeline.monitor_telemetry_batch(healthy_batch, step=1, run_id=run_id)
    assert not mon_res_healthy["drift_result"]["dataset_drift"]
    assert not mon_res_healthy["retraining_decision"]["trigger_retraining"]

    # Step 3: Stream fault batch (Bearing Wear) -> verify drift detected & retraining triggered
    fault_batch = features[features["fault_class"] == 1].iloc[:15]
    mon_res_fault = pipeline.monitor_telemetry_batch(fault_batch, step=2, run_id=run_id)
    assert mon_res_fault["drift_result"]["dataset_drift"]
    assert mon_res_fault["retraining_decision"]["trigger_retraining"]
    assert mon_res_fault["mlflow_logged"]

    # Step 4: Closed-loop retraining trigger execution
    retrained_engine, retrain_run_id = pipeline.trigger_retraining_workflow(
        trigger_reason=mon_res_fault["retraining_decision"]["reason"],
        features_df=features,
        sequences_x=sequences_x,
        classes_y=classes_y,
        epochs=2,
    )
    assert retrained_engine is not None
    assert retrain_run_id != run_id
