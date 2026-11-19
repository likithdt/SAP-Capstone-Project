"""
Simulated Dataset Generator for Industrial IoT Digital Twin & Predictive Maintenance.
Generates comprehensive synthetic telemetry datasets across all 7 operational fault classes:
  - Class 0: Healthy Baseline (1800 RPM, 48°C, 1.2A, low vibration)
  - Class 1: Bearing Wear (BPFO at 107.4 Hz, peak vibration, high crest factor)
  - Class 2: Rotor Imbalance (1X at 30 Hz sinusoidal dominant peak)
  - Class 3: Shaft Misalignment (2X at 60 Hz and 3X at 90 Hz harmonics)
  - Class 4: Thermal Overheating (elevated temp 95-115°C, thermal drift)
  - Class 5: Motor Overload (100 Hz ripple, current spikes)
  - Class 6: Sensor Anomaly (random spikes/dropouts, transient glitches)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.config import AssetConfig, FaultClass, DEFAULT_ASSET_CONFIG
from src.digital_twin.simulator import PhysicalSignalSimulator, TelemetryBatch


def generate_full_dataset(
    output_dir: str = "data",
    samples_per_class: int = 50,
    duration_sec: float = 0.5,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Generate complete simulated dataset files in CSV and JSON formats.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scenarios_dir = out_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    simulator = PhysicalSignalSimulator(DEFAULT_ASSET_CONFIG)
    
    history_records: List[Dict[str, Any]] = []
    batches_data: List[Dict[str, Any]] = []
    
    base_time = datetime.now(timezone.utc) - timedelta(minutes=samples_per_class * 7 * 2)

    # Progression order for a realistic lifecycle timeline
    # 1. Healthy (normal operation)
    # 2. Rotor Imbalance (early unbalance)
    # 3. Shaft Misalignment (mechanical looseness)
    # 4. Bearing Wear (BPFO degradation)
    # 5. Motor Overload (load spike)
    # 6. Thermal Overheating (critical thermal runaway)
    # 7. Sensor Anomaly (isolated electrical glitch)
    fault_sequence = [
        (FaultClass.HEALTHY, "Nominal Healthy", "RUNNING", "HEALTHY", 80),
        (FaultClass.ROTOR_IMBALANCE, "Rotor Imbalance (1X)", "RUNNING", "WATCHLIST", 40),
        (FaultClass.SHAFT_MISALIGNMENT, "Shaft Misalignment (2X/3X)", "RUNNING", "WARNING", 40),
        (FaultClass.BEARING_WEAR, "Bearing Wear (BPFO)", "RUNNING", "CRITICAL", 60),
        (FaultClass.MOTOR_OVERLOAD, "Motor Overload", "WARNING", "CRITICAL", 40),
        (FaultClass.THERMAL_OVERHEATING, "Thermal Overheating", "ALERT", "MAINTENANCE", 50),
        (FaultClass.SENSOR_ANOMALY, "Sensor Anomaly", "GLITCH", "WATCHLIST", 30),
    ]

    total_idx = 0
    current_time = base_time

    for fc, fault_name, motor_status, maintenance_status, count in fault_sequence:
        for i in range(count):
            current_time += timedelta(seconds=random.uniform(5, 15))
            
            # Generate waveform window
            batch: TelemetryBatch = simulator.generate_window(
                fault_class=fc,
                duration_sec=duration_sec,
                seed=seed + total_idx,
            )
            
            # Extract frequency metrics
            freqs, fft_amps = batch.compute_fft("accel_x")
            top_freq_idx = np.argsort(fft_amps)[-3:]
            dominant_freqs = [round(float(freqs[idx]), 1) for idx in reversed(top_freq_idx)]
            
            # Add synthetic humidity (35% - 75%)
            base_humidity = 55.0 - (batch.mean_temperature - 50.0) * 0.4
            humidity = round(max(20.0, min(85.0, base_humidity + random.uniform(-3.0, 3.0))), 2)

            record = {
                "timestamp": current_time.isoformat(timespec="milliseconds"),
                "deviceId": batch.device_id,
                "asset_id": batch.asset_id,
                "temperature": round(batch.mean_temperature, 2),
                "vibration": round(batch.rms_vibration, 3),
                "peak_vibration": round(batch.peak_vibration, 3),
                "crest_factor": round(batch.crest_factor, 2),
                "kurtosis": round(batch.kurtosis, 2),
                "current": round(batch.mean_current, 3),
                "speed_rpm": round(batch.mean_speed_rpm, 1),
                "humidity": humidity,
                "motorStatus": motor_status,
                "maintenanceStatus": maintenance_status,
                "fault_class": int(fc),
                "fault_name": fault_name,
                "dominant_freq_hz": dominant_freqs[0] if dominant_freqs else 30.0,
                "ingestedAt": current_time.isoformat(timespec="milliseconds"),
            }
            history_records.append(record)

            # Store raw waveform batch (subsampled for compactness)
            subsample = max(1, len(batch.accel_x) // 100)
            batch_record = {
                "batch_id": f"BATCH-{total_idx:04d}",
                "timestamp": current_time.isoformat(timespec="milliseconds"),
                "asset_id": batch.asset_id,
                "device_id": batch.device_id,
                "fault_class": int(fc),
                "fault_name": fault_name,
                "temperature_c": round(batch.mean_temperature, 2),
                "current_a": round(batch.mean_current, 3),
                "rms_vibration": round(batch.rms_vibration, 3),
                "speed_rpm": round(batch.mean_speed_rpm, 1),
                "vibration_x": [round(float(v), 4) for v in batch.accel_x[::subsample]],
                "vibration_y": [round(float(v), 4) for v in batch.accel_y[::subsample]],
                "vibration_z": [round(float(v), 4) for v in batch.accel_z[::subsample]],
            }
            batches_data.append(batch_record)
            total_idx += 1

    # 1. Save History DataFrame as CSV
    df = pd.DataFrame(history_records)
    csv_path = out_path / "simulated_telemetry_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records -> {csv_path}")

    # 2. Save History as JSON
    json_path = out_path / "simulated_telemetry_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)
    print(f"Generated JSON history -> {json_path}")

    # 3. Save Batches as JSON
    batches_path = out_path / "simulated_telemetry_batch.json"
    with open(batches_path, "w", encoding="utf-8") as f:
        json.dump(batches_data, f, indent=2)
    print(f"Generated {len(batches_data)} waveform batches -> {batches_path}")

    # 4. Generate Dedicated Scenario Presets
    scenario_map = {
        "scenario_1_healthy.json": {
            "name": "Scenario 1: Nominal Healthy Equipment",
            "fault_class": int(FaultClass.HEALTHY),
            "description": "Asset operates stably at 1800 RPM, 49.5°C, 1.22A, vibration < 0.3g. Zero false alarms.",
            "sample_record": [r for r in history_records if r["fault_class"] == 0][:10],
            "expected_state": "HEALTHY",
            "expected_work_order": None,
        },
        "scenario_2_bearing_defect.json": {
            "name": "Scenario 2: Developing Bearing Defect (BPFO)",
            "fault_class": int(FaultClass.BEARING_WEAR),
            "description": "Outer race defect producing 107.4 Hz impact frequency, vibration > 2.0g. Dispatches SAP PM Work Order.",
            "sample_record": [r for r in history_records if r["fault_class"] == 1][:10],
            "expected_state": "MAINTENANCE",
            "expected_work_order": "SAP PM Priority 2 Work Order",
        },
        "scenario_3_thermal_stockout.json": {
            "name": "Scenario 3: Severe Thermal Overheating & Stockout",
            "fault_class": int(FaultClass.THERMAL_OVERHEATING),
            "description": "Temperature drifts past 100°C. Bearing replacement spare part out of stock; triggers automated SAP MM Purchase Requisition.",
            "sample_record": [r for r in history_records if r["fault_class"] == 4][:10],
            "expected_state": "MAINTENANCE",
            "expected_work_order": "SAP PM Priority 1 Work Order + SAP MM Purchase Requisition",
        },
        "scenario_4_sensor_glitch.json": {
            "name": "Scenario 4: Transient Sensor Glitch Discrimination",
            "fault_class": int(FaultClass.SENSOR_ANOMALY),
            "description": "High vibration spike without temperature or current rise. Diagnosed as sensor anomaly; unnecessary mechanical work order avoided.",
            "sample_record": [r for r in history_records if r["fault_class"] == 6][:10],
            "expected_state": "WATCHLIST",
            "expected_work_order": "Sensor calibration ticket (zero mechanical false alarms)",
        },
    }

    for fname, sc_data in scenario_map.items():
        sc_file = scenarios_dir / fname
        with open(sc_file, "w", encoding="utf-8") as f:
            json.dump(sc_data, f, indent=2)
        print(f"Generated scenario preset -> {sc_file}")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "batches": str(batches_path),
        "scenarios_dir": str(scenarios_dir),
        "record_count": len(df),
    }


if __name__ == "__main__":
    generate_full_dataset()
