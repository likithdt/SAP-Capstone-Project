"""
Unit Tests for Deep Learning Prognostics Portfolio (Step 3).
Validates Isolation Forest, LSTM Autoencoder, 1D-CNN Fault Classifier, CNN-LSTM RUL Regressor,
and the Unified PrognosticsEngine.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import AssetConfig, FaultClass
from src.digital_twin.simulator import PhysicalSignalSimulator
from src.features.pipeline import FeatureExtractionPipeline
from src.models.isolation_forest import IsolationForestBaseline
from src.models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderDetector
from src.models.cnn_classifier import FaultClassifier1DCNN, FaultClassifier
from src.models.cnn_lstm_regressor import CNNLSTMRegressor, RULPredictor
from src.models.engine import PrognosticsEngine, DiagnosticReport


@pytest.fixture
def simulator():
    return PhysicalSignalSimulator()


@pytest.fixture
def feature_pipeline():
    return FeatureExtractionPipeline()


# -----------------------------------------------------------------------------
# 1. Isolation Forest Unit Tests
# -----------------------------------------------------------------------------
def test_isolation_forest_baseline(simulator, feature_pipeline, tmp_path):
    """Test Isolation Forest training on healthy data and scoring anomalies."""
    # Generate healthy reference data
    healthy_batches = [simulator.generate_window(FaultClass.HEALTHY, duration_sec=0.5, seed=s) for s in range(25)]
    df_healthy = feature_pipeline.transform_batches(healthy_batches)

    iso = IsolationForestBaseline(contamination=0.08, random_state=42)
    iso.fit(df_healthy)
    assert iso.is_fitted

    # Score healthy sample: should be low anomaly score
    h_test = simulator.generate_window(FaultClass.HEALTHY, duration_sec=0.5, seed=999)
    res_h = iso.score_single(feature_pipeline.extract_from_batch(h_test))
    assert res_h["anomaly_score"] < 0.60

    # Score severe anomaly (bearing wear / overload): should have higher score
    fault_test = simulator.generate_window(FaultClass.MOTOR_OVERLOAD, duration_sec=0.5, seed=123)
    res_f = iso.score_single(feature_pipeline.extract_from_batch(fault_test))
    assert res_f["anomaly_score"] > res_h["anomaly_score"]

    # Test serialization
    model_path = str(tmp_path / "iso_forest.joblib")
    iso.save(model_path)
    loaded = IsolationForestBaseline.load(model_path)
    assert loaded.is_fitted
    res_loaded = loaded.score_single(feature_pipeline.extract_from_batch(h_test))
    assert res_loaded["anomaly_score"] == res_h["anomaly_score"]


# -----------------------------------------------------------------------------
# 2. LSTM Autoencoder Unit Tests
# -----------------------------------------------------------------------------
def test_lstm_autoencoder_forward_and_reconstruction():
    """Verify PyTorch tensor shapes and forward pass of LSTM Autoencoder."""
    batch_size = 4
    seq_len = 100
    channels = 3

    model = LSTMAutoencoder(in_channels=channels, enc_hidden_1=32, latent_dim=16, dec_hidden_1=16, dec_hidden_2=32)
    x = torch.randn(batch_size, seq_len, channels)
    out = model(x)
    assert out.shape == (batch_size, seq_len, channels)


def test_lstm_autoencoder_training_and_anomaly(simulator, tmp_path):
    """Train autoencoder on healthy windows and test error difference on faults."""
    # 15 healthy windows of 200 samples (0.2s duration to keep test fast)
    healthy_data = []
    for s in range(15):
        b = simulator.generate_window(FaultClass.HEALTHY, duration_sec=0.2, seed=s)
        # (seq_len, 3)
        triaxial = np.stack([b.accel_x, b.accel_y, b.accel_z], axis=1)
        healthy_data.append(triaxial)
    healthy_arr = np.array(healthy_data)  # (15, 200, 3)

    detector = LSTMAutoencoderDetector(in_channels=3)
    detector.fit(healthy_arr, epochs=8, batch_size=8, lr=0.01)
    assert detector.is_fitted
    assert detector.reconstruction_threshold > 0.0

    # Healthy test window error
    h_test = simulator.generate_window(FaultClass.HEALTHY, duration_sec=0.2, seed=77)
    h_tri = np.stack([h_test.accel_x, h_test.accel_y, h_test.accel_z], axis=1)
    h_res = detector.score_single_window(h_tri)

    # Fault test window (Imbalance: high 1X amplitude)
    f_test = simulator.generate_window(FaultClass.ROTOR_IMBALANCE, duration_sec=0.2, seed=77)
    f_tri = np.stack([f_test.accel_x, f_test.accel_y, f_test.accel_z], axis=1)
    f_res = detector.score_single_window(f_tri)

    # Reconstruction error on imbalance must be noticeably higher than healthy
    assert f_res["reconstruction_error"] > h_res["reconstruction_error"]

    # Test save and load
    save_path = str(tmp_path / "lstm_ae.pt")
    detector.save(save_path)
    loaded = LSTMAutoencoderDetector.load(save_path)
    assert loaded.is_fitted
    loaded_res = loaded.score_single_window(h_tri)
    assert abs(loaded_res["reconstruction_error"] - h_res["reconstruction_error"]) < 1e-4


# -----------------------------------------------------------------------------
# 3. 1D-CNN Fault Classifier Unit Tests
# -----------------------------------------------------------------------------
def test_1d_cnn_classifier_forward():
    """Verify PyTorch tensor shapes and softmax output of 1D-CNN."""
    batch_size = 4
    channels = 3
    seq_len = 200
    num_classes = 7

    model = FaultClassifier1DCNN(in_channels=channels, num_classes=num_classes)
    x = torch.randn(batch_size, channels, seq_len)
    logits = model(x)
    assert logits.shape == (batch_size, num_classes)


def test_1d_cnn_classifier_train_and_predict(simulator, tmp_path):
    """Train 1D-CNN on synthetic windows and test prediction."""
    X_list = []
    y_list = []
    # 2 samples for each class, length 200
    for fc in FaultClass:
        for s in range(2):
            b = simulator.generate_window(fc, duration_sec=0.2, seed=s*10 + fc.value)
            channels = np.stack([b.accel_x, b.accel_y, b.accel_z], axis=0)  # (3, 200)
            X_list.append(channels)
            y_list.append(fc.value)

    X_train = np.array(X_list)
    y_train = np.array(y_list)

    classifier = FaultClassifier(in_channels=3, num_classes=7)
    classifier.fit(X_train, y_train, epochs=5, batch_size=7, lr=0.005)
    assert classifier.is_fitted

    pred_res = classifier.predict_single(X_train[0])
    assert "predicted_class" in pred_res
    assert "predicted_label" in pred_res
    assert "confidence" in pred_res
    assert len(pred_res["class_probabilities"]) == 7

    # Test save and load
    save_path = str(tmp_path / "cnn_classifier.pt")
    classifier.save(save_path)
    loaded = FaultClassifier.load(save_path)
    assert loaded.is_fitted
    loaded_res = loaded.predict_single(X_train[0])
    assert loaded_res["predicted_class"] == pred_res["predicted_class"]


# -----------------------------------------------------------------------------
# 4. CNN-LSTM RUL Regressor Unit Tests
# -----------------------------------------------------------------------------
def test_cnn_lstm_regressor_forward():
    """Verify forward pass of CNN-LSTM RUL network."""
    batch_size = 4
    channels = 3
    seq_len = 200

    model = CNNLSTMRegressor(in_channels=channels)
    x = torch.randn(batch_size, channels, seq_len)
    rul = model(x)
    assert rul.shape == (batch_size, 1)


def test_cnn_lstm_regressor_train_and_predict(tmp_path):
    """Train RUL predictor on dummy degradation sequences."""
    X = np.random.randn(8, 3, 100).astype(np.float32)
    y = np.array([100.0, 85.0, 70.0, 55.0, 40.0, 25.0, 10.0, 2.0], dtype=np.float32)

    predictor = RULPredictor(in_channels=3, max_rul_hours=120.0)
    predictor.fit(X, y, epochs=5, batch_size=4, lr=0.01)
    assert predictor.is_fitted

    rul_single = predictor.predict_single(X[0])
    assert 0.0 <= rul_single <= 120.0

    # Test save and load
    save_path = str(tmp_path / "rul_model.pt")
    predictor.save(save_path)
    loaded = RULPredictor.load(save_path)
    assert loaded.is_fitted
    assert loaded.predict_single(X[0]) == rul_single


# -----------------------------------------------------------------------------
# 5. PrognosticsEngine Integration Tests
# -----------------------------------------------------------------------------
def test_prognostics_engine_healthy(simulator):
    """Verify end-to-end inference on healthy asset window."""
    engine = PrognosticsEngine()
    batch = simulator.generate_window(FaultClass.HEALTHY, duration_sec=1.0, seed=42)

    report = engine.predict_batch(batch)
    assert isinstance(report, DiagnosticReport)
    assert report.asset_id == "MOTOR-001"
    assert report.predicted_class == 0
    assert report.predicted_failure_mode == "Healthy"
    assert report.risk_level == "LOW"
    assert report.anomaly_score < 0.35
    assert not report.is_anomaly
    assert report.rul_hours >= 100.0
    assert len(report.evidence) > 0
    assert "Normal operation" in report.recommended_action


def test_prognostics_engine_bearing_wear(simulator):
    """Verify end-to-end inference on bearing wear (BPFO)."""
    engine = PrognosticsEngine()
    batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=1.0, seed=42)

    report = engine.predict_batch(batch)
    assert report.predicted_class == 1
    assert "Bearing Wear" in report.predicted_failure_mode
    assert report.is_anomaly
    assert report.rul_hours <= 48.0
    assert any("Kurtosis" in ev or "BPFO" in ev for ev in report.evidence)
    assert "6205-2RS Bearing" in report.recommended_action


def test_prognostics_engine_shaft_misalignment(simulator):
    """Verify end-to-end inference on shaft misalignment."""
    engine = PrognosticsEngine()
    batch = simulator.generate_window(FaultClass.SHAFT_MISALIGNMENT, duration_sec=1.0, seed=42)

    report = engine.predict_batch(batch)
    assert report.predicted_class == 3
    assert "Misalignment" in report.predicted_failure_mode
    assert report.is_anomaly
    assert any("axial" in ev.lower() or "2X" in ev for ev in report.evidence)
    assert "laser alignment" in report.recommended_action.lower()


def test_prognostics_engine_thermal_overheating(simulator):
    """Verify end-to-end inference on thermal overheating."""
    engine = PrognosticsEngine()
    batch = simulator.generate_window(FaultClass.THERMAL_OVERHEATING, duration_sec=1.0, seed=42)

    report = engine.predict_batch(batch)
    assert report.predicted_class == 4
    assert "Thermal Overheating" in report.predicted_failure_mode
    assert report.is_anomaly
    assert report.risk_level in ["HIGH", "CRITICAL"]
    assert any("thermal" in ev.lower() or "85" in ev for ev in report.evidence)


def test_prognostics_engine_motor_overload(simulator):
    """Verify end-to-end inference on motor overload."""
    engine = PrognosticsEngine()
    batch = simulator.generate_window(FaultClass.MOTOR_OVERLOAD, duration_sec=1.0, seed=42)

    report = engine.predict_batch(batch)
    assert report.predicted_class == 5
    assert "Motor Overload" in report.predicted_failure_mode
    assert report.is_anomaly
    assert any("current" in ev.lower() or "100" in ev for ev in report.evidence)


def test_prognostics_engine_sensor_anomaly(simulator):
    """Verify end-to-end inference on sensor anomaly (glitches/spikes)."""
    engine = PrognosticsEngine()
    batch = simulator.generate_window(FaultClass.SENSOR_ANOMALY, duration_sec=1.0, seed=42, allow_nan=True)

    report = engine.predict_batch(batch)
    assert report.predicted_class == 6
    assert "Sensor Anomaly" in report.predicted_failure_mode
    assert report.is_anomaly
    assert any("sensor" in ev.lower() or "spikes" in ev.lower() for ev in report.evidence)
    assert "wiring" in report.recommended_action.lower() or "mounting" in report.recommended_action.lower()


def test_prognostics_engine_save_and_load(simulator, tmp_path):
    """Verify serialization and deserialization of complete PrognosticsEngine."""
    engine = PrognosticsEngine()
    batch = simulator.generate_window(FaultClass.HEALTHY, duration_sec=1.0, seed=42)
    rep1 = engine.predict_batch(batch)

    save_dir = str(tmp_path / "engine_bundle")
    engine.save(save_dir)

    loaded_engine = PrognosticsEngine.load(save_dir)
    rep2 = loaded_engine.predict_batch(batch)

    assert rep1.predicted_class == rep2.predicted_class
    assert rep1.predicted_failure_mode == rep2.predicted_failure_mode
    assert abs(rep1.anomaly_score - rep2.anomaly_score) < 1e-4

