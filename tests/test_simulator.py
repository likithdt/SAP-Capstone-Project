"""
Unit Tests for Physical Signal Simulator (Step 1).
Verifies all 7 operational classes against physical signal modeling specifications.
"""

import numpy as np
import pytest
from src.config import AssetConfig, FaultClass, ClassThresholds
from src.digital_twin.simulator import PhysicalSignalSimulator, TelemetryBatch


@pytest.fixture
def simulator():
    config = AssetConfig(
        nominal_speed_rpm=1800.0,
        sampling_rate_hz=1000.0,
        line_frequency_hz=50.0,
        bpfo_factor=3.58,
    )
    return PhysicalSignalSimulator(config)


def test_asset_config_properties(simulator):
    """Verify mathematical derived parameters of the asset."""
    assert simulator.nominal_rpm == 1800.0
    assert simulator.fs == 1000.0
    assert simulator.f0 == 30.0  # 1800 / 60
    assert abs(simulator.bpfo - 107.4) < 1e-4  # 3.58 * 30 Hz
    assert simulator.f_line == 50.0


def test_class_0_healthy_specifications(simulator):
    """
    Class 0 (Healthy):
      - Low-amplitude fundamental vibration at f0 (0.15–0.35 g RMS)
      - Steady baseline temperature (45–52 °C)
      - Nominal idle current (1.1–1.3 A)
    """
    for seed in [10, 42, 99]:
        batch = simulator.generate_window(FaultClass.HEALTHY, duration_sec=1.0, seed=seed)
        
        # 1. RMS vibration check [0.15, 0.35] g
        assert 0.15 <= batch.rms_vibration <= 0.35, f"Healthy RMS {batch.rms_vibration} outside [0.15, 0.35] g"
        
        # 2. Temperature check [45, 52] °C
        assert np.all(batch.temperature_c >= 45.0), "Temperature below 45°C"
        assert np.all(batch.temperature_c <= 52.0), "Temperature above 52°C"
        assert 45.0 <= batch.mean_temperature <= 52.0
        
        # 3. Current check [1.1, 1.3] A
        assert 1.10 <= batch.mean_current <= 1.30, f"Healthy current {batch.mean_current} outside [1.1, 1.3] A"
        
        # 4. Dominant frequency should be at fundamental 1X (30 Hz)
        assert abs(batch.dominant_frequency("accel_x") - 30.0) <= 2.0


def test_class_1_bearing_wear_bpfo(simulator):
    """
    Class 1 (Bearing Wear):
      - Periodic impact impulses at BPFO (~107.4 Hz)
      - High kurtosis (> 4.5)
      - Elevated crest factor (> 4.0)
    """
    for seed in [15, 42, 101]:
        batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=1.0, seed=seed)
        
        # 1. Kurtosis check (> 4.5)
        assert batch.kurtosis > ClassThresholds.BEARING_KURTOSIS_THRESHOLD, (
            f"Bearing wear kurtosis {batch.kurtosis} not > 4.5"
        )
        
        # 2. Elevated crest factor (> 4.0)
        assert batch.crest_factor >= ClassThresholds.BEARING_CREST_FACTOR_MIN, (
            f"Bearing wear crest factor {batch.crest_factor} not >= 4.0"
        )
        
        # 3. BPFO impact rate verification via envelope spectrum
        from scipy.signal import hilbert
        env = np.abs(hilbert(batch.accel_x))
        env = env - np.mean(env)
        freqs = np.fft.rfftfreq(len(env), d=1.0 / simulator.fs)
        fft_env = np.abs(np.fft.rfft(env))
        
        # Find peak around BPFO (107.4 Hz ± 5 Hz)
        bpfo_band = (freqs >= 102.0) & (freqs <= 113.0)
        max_in_band = np.max(fft_env[bpfo_band])
        median_out_band = np.median(fft_env[freqs > 10.0])
        assert max_in_band > 2.0 * median_out_band, "Envelope spectrum should exhibit pronounced BPFO peak"


def test_class_2_rotor_imbalance_1x_scaling(simulator):
    """
    Class 2 (Rotor Imbalance):
      - High-amplitude sinusoidal vibration centered strictly at fundamental 1X (30 Hz)
      - Vibration energy scaling with rotational speed (N/1800)^2
    """
    # 1. Imbalance at nominal speed (1800 RPM)
    batch_1800 = simulator.generate_window(FaultClass.ROTOR_IMBALANCE, speed_rpm=1800.0, seed=42)
    assert batch_1800.dominant_frequency("accel_x") == 30.0
    assert batch_1800.rms_vibration >= ClassThresholds.IMBALANCE_RMS_MIN_G
    
    # Check that 1X power dominates spectrum (> 70% of spectral energy)
    freqs, amps = batch_1800.compute_fft("accel_x")
    idx_1x = np.argmin(np.abs(freqs - 30.0))
    p_1x = amps[idx_1x]**2
    p_total = np.sum(amps[1:]**2)
    assert (p_1x / p_total) > ClassThresholds.IMBALANCE_1X_POWER_RATIO_MIN

    # 2. Speed scaling test: compare 1200 RPM vs 2400 RPM
    batch_1200 = simulator.generate_window(FaultClass.ROTOR_IMBALANCE, speed_rpm=1200.0, seed=42)
    batch_2400 = simulator.generate_window(FaultClass.ROTOR_IMBALANCE, speed_rpm=2400.0, seed=42)
    
    # (2400 / 1200)^2 = 4x power scaling in amplitude
    ratio = batch_2400.rms_vibration / batch_1200.rms_vibration
    assert 3.2 <= ratio <= 4.8, f"Imbalance vibration did not scale quadratically with speed: ratio={ratio:.2f}"


def test_class_3_shaft_misalignment_harmonics(simulator):
    """
    Class 3 (Shaft Misalignment):
      - Strong harmonic peaks at 2X (60 Hz) and 3X (90 Hz)
      - Coupled with moderate phase offset and axial vibration
    """
    batch = simulator.generate_window(FaultClass.SHAFT_MISALIGNMENT, duration_sec=1.0, seed=42)
    
    freqs, amps = batch.compute_fft("accel_x")
    amp_1x = amps[np.argmin(np.abs(freqs - 30.0))]
    amp_2x = amps[np.argmin(np.abs(freqs - 60.0))]
    amp_3x = amps[np.argmin(np.abs(freqs - 90.0))]
    
    # 2X harmonic should be prominent (> 50% of 1X or greater)
    assert amp_2x > ClassThresholds.MISALIGNMENT_2X_HARMONIC_MIN_RATIO * amp_1x
    assert amp_3x > 0.3 * amp_1x
    
    # Axial vibration (accel_z) must be significant compared to radial
    rms_radial = batch.rms_vibration
    rms_axial = np.sqrt(np.mean(batch.accel_z**2))
    assert (rms_axial / rms_radial) >= ClassThresholds.MISALIGNMENT_AXIAL_RATIO_MIN


def test_class_4_thermal_overheating_drift(simulator):
    """
    Class 4 (Thermal Overheating):
      - Progressive thermal drift from ~55 °C climbing beyond 85 °C critical threshold
      - Mild amplitude modulation due to thermal expansion
    """
    batch = simulator.generate_window(FaultClass.THERMAL_OVERHEATING, duration_sec=1.0, seed=42)
    
    # Start temperature near 55 °C
    assert abs(batch.temperature_c[0] - ClassThresholds.THERMAL_OVERHEATING_START_C) <= 2.5
    # End temperature exceeding 85 °C critical threshold
    assert batch.temperature_c[-1] > ClassThresholds.THERMAL_OVERHEATING_CRITICAL_C
    
    # Monotonic drift check
    assert batch.temperature_c[-1] > batch.temperature_c[0] + 25.0
    
    # Amplitude modulation: latter half vibration amplitude should be higher than beginning
    half = len(batch.accel_x) // 2
    rms_first_half = np.sqrt(np.mean(batch.accel_x[:half]**2))
    rms_second_half = np.sqrt(np.mean(batch.accel_x[half:]**2))
    assert rms_second_half > rms_first_half


def test_class_5_motor_overload_current_and_ripple(simulator):
    """
    Class 5 (Motor Overload):
      - High current draw (2.4–3.8 A vs. 1.2 A nominal)
      - Frequency line ripples at 2 * 50 Hz = 100 Hz
      - Lagging thermal rise
    """
    batch = simulator.generate_window(FaultClass.MOTOR_OVERLOAD, duration_sec=1.0, seed=42)
    
    # 1. High current draw [2.4, 3.8] A
    assert ClassThresholds.OVERLOAD_CURRENT_MIN_A <= batch.mean_current <= ClassThresholds.OVERLOAD_CURRENT_MAX_A
    
    # 2. 100 Hz line ripple in current
    curr_ac = batch.current_a - np.mean(batch.current_a)
    freqs, amps = np.fft.rfftfreq(len(curr_ac), 1.0 / simulator.fs), np.abs(np.fft.rfft(curr_ac))
    top_curr_freq = freqs[np.argmax(amps)]
    assert abs(top_curr_freq - ClassThresholds.OVERLOAD_LINE_RIPPLE_FREQ_HZ) <= 2.0
    
    # 3. Lagging thermal rise: temperature at end is higher than at start
    assert batch.temperature_c[-1] > batch.temperature_c[0] + 5.0


def test_class_6_sensor_anomaly_dropouts_and_spikes(simulator):
    """
    Class 6 (Sensor Anomaly):
      - Signal dropouts (zero-hold/flatlines)
      - Random float spikes (> 15 g or NaN)
      - White noise saturation
    """
    batch = simulator.generate_window(FaultClass.SENSOR_ANOMALY, duration_sec=1.0, seed=42, allow_nan=True)
    
    # 1. Spikes exceeding 15 g
    max_spike = np.nanmax(np.abs(batch.accel_x))
    assert max_spike > ClassThresholds.SENSOR_SPIKE_THRESHOLD_G
    
    # 2. Signal dropouts (consecutive zeros)
    zeros = np.sum(batch.accel_x == 0.0)
    assert zeros >= 50, "Should contain at least 50 ms of dropout zeros"
    
    # 3. NaNs present when allow_nan=True
    assert np.isnan(batch.accel_x).any()
    assert np.isnan(batch.temperature_c).any()


def test_telemetry_serialization_and_mqtt(simulator):
    """Verify conversion to DataFrame and AWS IoT Core MQTT payload schema."""
    batch = simulator.generate_window(FaultClass.HEALTHY, duration_sec=1.0, seed=42)
    
    # DataFrame conversion
    df = batch.to_dataframe()
    assert len(df) == 1000
    assert "accel_x" in df.columns
    assert "temperature_c" in df.columns
    assert "current_a" in df.columns
    assert "fault_class" in df.columns
    
    # MQTT payload serialization
    mqtt_msgs = batch.to_mqtt_payloads(sample_step=50)
    assert len(mqtt_msgs) == 20
    first = mqtt_msgs[0]
    
    required_keys = {
        "device_id", "asset_id", "timestamp",
        "accel_x", "accel_y", "accel_z",
        "temperature_c", "current_a", "rms_vibration",
    }
    assert required_keys.issubset(first.keys())
    assert first["device_id"] == "ESP32-MOTOR-001"
    assert first["asset_id"] == "MOTOR-001"


def test_dataset_generation(simulator, tmp_path):
    """Verify synthetic dataset generator across all 7 classes."""
    out_dir = str(tmp_path / "raw_data")
    df = simulator.generate_dataset(samples_per_class=5, duration_sec=0.5, seed=42, output_dir=out_dir)
    
    # 7 classes * 5 samples = 35 rows
    assert len(df) == 35
    assert set(df["fault_class"].unique()) == set(range(7))
    assert "rms_g" in df.columns
    assert "kurtosis" in df.columns
    assert "dominant_freq_hz" in df.columns
