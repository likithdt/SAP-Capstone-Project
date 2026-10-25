"""
Unit Tests for Feature Engineering Pipeline (Step 2).
Validates time-domain statistics, FFT spectral analysis, envelope demodulation,
cross-channel metrics, and pipeline transformations across all fault classes.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import AssetConfig, FaultClass
from src.digital_twin.simulator import PhysicalSignalSimulator
from src.features.time_domain import (
    compute_time_domain_features,
    compute_temperature_features,
    compute_current_features,
)
from src.features.frequency_domain import (
    compute_rfft_spectrum,
    compute_spectral_moments,
    compute_spectral_entropy_and_flatness,
    compute_band_energies,
    compute_envelope_bpfo_features,
    compute_frequency_domain_features,
)
from src.features.pipeline import FeatureExtractionPipeline


# -----------------------------------------------------------------------------
# 1. Time-Domain Analytical Unit Tests
# -----------------------------------------------------------------------------
def test_time_domain_sine_wave_analytical():
    """Verify analytical metrics for a known pure sine wave: A * sin(wt)."""
    fs = 1000.0
    t = np.arange(1000) / fs
    amp = 2.0
    sig = amp * np.sin(2 * np.pi * 30.0 * t)

    feats = compute_time_domain_features(sig, prefix="test")

    # Analytical: Peak = 2.0, RMS = 2.0 / sqrt(2) ≈ 1.4142
    assert abs(feats["test_peak"] - 2.0) < 1e-3
    assert abs(feats["test_rms"] - (2.0 / np.sqrt(2))) < 1e-2
    assert abs(feats["test_crest_factor"] - np.sqrt(2)) < 1e-2
    # Sine wave kurtosis is strictly 1.5 (Pearson)
    assert abs(feats["test_kurtosis"] - 1.5) < 0.1
    # Symmetric sine wave skewness is 0
    assert abs(feats["test_skewness"]) < 0.05
    # Zero crossing rate for 30 Hz sine wave sampled at 1000 Hz: 2 * 30 / 1000 = 0.06
    assert abs(feats["test_zero_crossing_rate"] - 0.06) < 0.01


def test_time_domain_gaussian_noise():
    """Verify kurtosis of Gaussian noise is close to 3.0."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, 10000)
    feats = compute_time_domain_features(noise, prefix="noise")
    assert abs(feats["noise_kurtosis"] - 3.0) < 0.15
    assert abs(feats["noise_mean"]) < 0.05


def test_temperature_features():
    """Verify thermal drift slope extraction."""
    # Ramp from 50 to 90 over 1000 steps
    temp_signal = np.linspace(50.0, 90.0, 1000)
    feats = compute_temperature_features(temp_signal)
    assert feats["temp_min"] == 50.0
    assert feats["temp_max"] == 90.0
    assert abs(feats["temp_mean"] - 70.0) < 1e-3
    assert feats["temp_slope"] > 0.035  # Positive drift


def test_current_features():
    """Verify current mean, RMS, and 100 Hz ripple detection."""
    fs = 1000.0
    t = np.arange(1000) / fs
    current = 3.0 + 0.4 * np.sin(2 * np.pi * 100.0 * t)  # 3A DC + 0.4A AC ripple at 100 Hz
    feats = compute_current_features(current, fs=fs)
    assert abs(feats["current_mean"] - 3.0) < 0.02
    assert abs(feats["current_100hz_ripple"] - 0.4) < 0.05


# -----------------------------------------------------------------------------
# 2. Frequency-Domain Analytical Unit Tests
# -----------------------------------------------------------------------------
def test_frequency_domain_spectral_peaks():
    """Verify FFT correctly identifies dominant frequencies and spectral moments."""
    fs = 1000.0
    t = np.arange(1000) / fs
    # Signal with 30 Hz fundamental (1.0g) + 60 Hz harmonic (0.5g)
    sig = 1.0 * np.sin(2 * np.pi * 30.0 * t) + 0.5 * np.sin(2 * np.pi * 60.0 * t)

    freqs, amps, power = compute_rfft_spectrum(sig, fs=fs)
    dom_idx = np.argmax(amps[1:]) + 1
    assert freqs[dom_idx] == 30.0

    bands = compute_band_energies(freqs, power, f0=30.0, bpfo=107.4)
    assert bands["band_1x_energy"] > 0
    assert bands["band_2x_energy"] > 0
    assert bands["band_1x_energy"] > bands["band_2x_energy"]

    # Moments
    moments = compute_spectral_moments(freqs, power)
    # Centroid should lie between 30 Hz and 60 Hz
    assert 30.0 <= moments["spectral_centroid"] <= 60.0


def test_envelope_bpfo_demodulation():
    """Verify Hilbert transform envelope spectrum extracts BPFO impacts correctly."""
    fs = 1000.0
    t = np.arange(1000) / fs
    # Amplitude-modulated carrier (320 Hz structural resonance modulated by 107.4 Hz impacts)
    carrier = np.sin(2 * np.pi * 340.0 * t)
    modulator = 1.0 + 0.8 * np.sin(2 * np.pi * 107.4 * t)
    sig = carrier * modulator

    env_feats = compute_envelope_bpfo_features(sig, fs=fs, bpfo=107.4)
    assert env_feats["envelope_bpfo_peak_amp"] > 0.3
    assert env_feats["envelope_bpfo_ratio"] > 3.0


# -----------------------------------------------------------------------------
# 3. Pipeline End-to-End Tests Across All 7 Simulator Fault Classes
# -----------------------------------------------------------------------------
@pytest.fixture
def pipeline():
    return FeatureExtractionPipeline()


@pytest.fixture
def simulator():
    return PhysicalSignalSimulator()


def test_pipeline_class_0_healthy(pipeline, simulator):
    batch = simulator.generate_window(FaultClass.HEALTHY, duration_sec=1.0, seed=42)
    feats = pipeline.extract_from_batch(batch)

    # Class 0 assertions
    assert 0.15 <= feats["rad_rms"] <= 0.35
    assert 45.0 <= feats["temp_mean"] <= 52.0
    assert 1.10 <= feats["current_mean"] <= 1.30
    assert feats["integrity_has_sensor_glitch"] == 0
    assert feats["temp_critical_flag"] == 0
    assert feats["current_overload_flag"] == 0


def test_pipeline_class_1_bearing_wear(pipeline, simulator):
    batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=1.0, seed=42)
    feats = pipeline.extract_from_batch(batch)

    # Class 1 assertions
    assert feats["rad_kurtosis"] > 4.5
    assert feats["rad_crest_factor"] >= 4.0
    assert feats["rad_envelope_bpfo_ratio"] > 1.5


def test_pipeline_class_2_rotor_imbalance(pipeline, simulator):
    batch = simulator.generate_window(FaultClass.ROTOR_IMBALANCE, duration_sec=1.0, seed=42)
    feats = pipeline.extract_from_batch(batch)

    # Class 2 assertions: 1X dominance
    assert feats["rad_dominant_freq_hz"] == 30.0
    assert feats["rad_rms"] > 0.8
    assert feats["rad_band_1x_energy"] > 5.0 * feats["rad_band_2x_energy"]


def test_pipeline_class_3_shaft_misalignment(pipeline, simulator):
    batch = simulator.generate_window(FaultClass.SHAFT_MISALIGNMENT, duration_sec=1.0, seed=42)
    feats = pipeline.extract_from_batch(batch)

    # Class 3 assertions: 2X/3X harmonics and axial-radial coupling
    assert feats["rad_band_2x_energy"] > 0.5 * feats["rad_band_1x_energy"]
    assert feats["axial_to_radial_rms_ratio"] > 0.5


def test_pipeline_class_4_thermal_overheating(pipeline, simulator):
    batch = simulator.generate_window(FaultClass.THERMAL_OVERHEATING, duration_sec=1.0, seed=42)
    feats = pipeline.extract_from_batch(batch)

    # Class 4 assertions: temperature climbs > 85°C
    assert feats["temp_critical_flag"] == 1
    assert feats["temp_max"] > 85.0
    assert feats["temp_slope"] > 0.02


def test_pipeline_class_5_motor_overload(pipeline, simulator):
    batch = simulator.generate_window(FaultClass.MOTOR_OVERLOAD, duration_sec=1.0, seed=42)
    feats = pipeline.extract_from_batch(batch)

    # Class 5 assertions: current draw > 2.4A and 100 Hz line ripple
    assert feats["current_overload_flag"] == 1
    assert feats["current_mean"] >= 2.4
    assert feats["current_100hz_ripple"] > 0.15


def test_pipeline_class_6_sensor_anomaly(pipeline, simulator):
    batch = simulator.generate_window(FaultClass.SENSOR_ANOMALY, duration_sec=1.0, seed=42, allow_nan=True)
    feats = pipeline.extract_from_batch(batch)

    # Class 6 assertions: integrity checks catch glitches
    assert feats["integrity_has_sensor_glitch"] == 1
    assert feats["integrity_nan_count"] > 0 or feats["integrity_extreme_spike_count"] > 0


def test_pipeline_batch_transform_to_dataframe(pipeline, simulator):
    """Verify batch processing into a multi-row feature DataFrame."""
    batches = [
        simulator.generate_window(FaultClass.HEALTHY, duration_sec=0.5, seed=1),
        simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=0.5, seed=2),
        simulator.generate_window(FaultClass.ROTOR_IMBALANCE, duration_sec=0.5, seed=3),
    ]

    df_features = pipeline.transform_batches(batches)

    assert isinstance(df_features, pd.DataFrame)
    assert len(df_features) == 3
    assert "rad_rms" in df_features.columns
    assert "rad_kurtosis" in df_features.columns
    assert "rad_dominant_freq_hz" in df_features.columns
    assert "temp_mean" in df_features.columns
    assert "current_mean" in df_features.columns
    assert "axial_to_radial_rms_ratio" in df_features.columns
    assert "fault_class" in df_features.columns
