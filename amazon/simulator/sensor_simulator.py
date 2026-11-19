"""
Sensor Simulator for Industrial Asset Telemetry.
Supports:
  1. Local Mode: Appends physics-simulated readings to data/simulated_telemetry_history.csv.
  2. API Mode: Posts telemetry batches to FastAPI Digital Twin server (http://127.0.0.1:8000/api/telemetry).
  3. MQTT Mode: Publishes encrypted TLS telemetry to AWS IoT Core MQTT broker.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

DATA_CSV_PATH = PROJECT_ROOT / "data" / "simulated_telemetry_history.csv"


def build_reading(device_id: str, asset_id: str = "MOTOR-001", fault_mode: str = "healthy") -> Dict[str, Any]:
    """
    Generate realistic multi-sensor telemetry based on selected fault profile.
    """
    now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    
    if fault_mode == "bearing":
        temp = random.uniform(53.0, 58.0)
        vib = random.uniform(2.1, 3.8)
        curr = random.uniform(1.30, 1.45)
        rpm = random.uniform(1796.0, 1800.0)
        dom_freq = 107.4
        fault_name = "Bearing Wear (BPFO)"
        fault_class = 1
        maint_status = "CRITICAL"
    elif fault_mode == "thermal":
        temp = random.uniform(96.0, 112.0)
        vib = random.uniform(0.8, 1.4)
        curr = random.uniform(1.80, 2.30)
        rpm = random.uniform(1790.0, 1795.0)
        dom_freq = 30.0
        fault_name = "Thermal Overheating"
        fault_class = 4
        maint_status = "MAINTENANCE"
    elif fault_mode == "imbalance":
        temp = random.uniform(51.0, 56.0)
        vib = random.uniform(1.2, 1.9)
        curr = random.uniform(1.28, 1.38)
        rpm = random.uniform(1798.0, 1801.0)
        dom_freq = 30.0
        fault_name = "Rotor Imbalance (1X)"
        fault_class = 2
        maint_status = "WATCHLIST"
    elif fault_mode == "misalignment":
        temp = random.uniform(55.0, 62.0)
        vib = random.uniform(1.0, 1.8)
        curr = random.uniform(1.40, 1.65)
        rpm = random.uniform(1797.0, 1800.0)
        dom_freq = 60.0
        fault_name = "Shaft Misalignment (2X/3X)"
        fault_class = 3
        maint_status = "WARNING"
    elif fault_mode == "glitch":
        temp = random.uniform(48.0, 51.0)
        vib = random.uniform(4.5, 8.0)  # Spike
        curr = random.uniform(1.18, 1.24)
        rpm = 1800.0
        dom_freq = 240.0
        fault_name = "Sensor Anomaly"
        fault_class = 6
        maint_status = "WATCHLIST"
    else:  # healthy
        temp = random.uniform(47.5, 50.5)
        vib = random.uniform(0.18, 0.28)
        curr = random.uniform(1.16, 1.23)
        rpm = random.uniform(1799.5, 1800.5)
        dom_freq = 30.0
        fault_name = "Nominal Healthy"
        fault_class = 0
        maint_status = "HEALTHY"

    humidity = round(max(20.0, min(80.0, 54.0 - (temp - 50.0) * 0.4 + random.uniform(-2, 2))), 1)

    return {
        "timestamp": now_ts,
        "deviceId": device_id,
        "asset_id": asset_id,
        "temperature": round(temp, 2),
        "vibration": round(vib, 3),
        "peak_vibration": round(vib * random.uniform(1.9, 2.3), 3),
        "crest_factor": round(random.uniform(1.9, 2.4), 2),
        "kurtosis": round(random.uniform(1.7, 2.2), 2),
        "current": round(curr, 3),
        "speed_rpm": round(rpm, 1),
        "humidity": humidity,
        "motorStatus": "RUNNING" if fault_mode != "thermal" else "ALERT",
        "maintenanceStatus": maint_status,
        "fault_class": fault_class,
        "fault_name": fault_name,
        "dominant_freq_hz": dom_freq,
        "ingestedAt": now_ts,
    }


def publish_local(reading: Dict[str, Any]):
    """Append reading to local dataset CSV."""
    import pandas as pd
    DATA_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([reading])
    if DATA_CSV_PATH.is_file():
        df_new.to_csv(DATA_CSV_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(DATA_CSV_PATH, index=False)
    print(f"[LOCAL] Stored: Temp={reading['temperature']}°C | Vib={reading['vibration']}g | Class={reading['fault_name']}")


def publish_api(reading: Dict[str, Any], api_url: str):
    """Post reading to FastAPI backend."""
    import requests
    payload = {
        "asset_id": reading["asset_id"],
        "device_id": reading["deviceId"],
        "temperature_c": reading["temperature"],
        "current_a": reading["current"],
        "rms_vibration": reading["vibration"],
        "speed_rpm": reading["speed_rpm"],
        "vibration_x": [float(reading["vibration"])] * 20,
    }
    try:
        resp = requests.post(api_url, json=payload, timeout=5)
        if resp.status_code == 200:
            res_data = resp.json()
            twin_info = res_data.get("twin_overview", {})
            print(f"[API] OK (200) -> Twin State={twin_info.get('state')} | Health={twin_info.get('health_score')}")
        else:
            print(f"[API] Response {resp.status_code}: {resp.text}")
    except Exception as exc:
        print(f"[API] Connection error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Simulate industrial IoT asset telemetry.")
    parser.add_argument("--mode", choices=["local", "api", "mqtt"], default=None, help="Telemetry publishing mode (default: auto-detect)")
    parser.add_argument("--fault", choices=["healthy", "bearing", "thermal", "imbalance", "misalignment", "glitch"], default="healthy", help="Fault condition to simulate")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between messages (default: 3)")
    parser.add_argument("--once", action="store_true", help="Publish single reading and exit")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api/telemetry", help="Target API URL for API mode")
    parser.add_argument("--abnormal", action="store_true", help="Legacy flag for abnormal reading (simulates bearing fault)")
    args = parser.parse_args()

    fault = "bearing" if args.abnormal else args.fault
    thing_name = os.getenv("AWS_IOT_THING_NAME", "ESP32-MOTOR-001")
    asset_id = "MOTOR-001"

    # Auto-detect mode if not specified
    mode = args.mode
    if mode is None:
        cert = os.getenv("AWS_IOT_CERT")
        endpoint = os.getenv("AWS_IOT_ENDPOINT")
        if cert and Path(cert).is_file() and endpoint:
            mode = "mqtt"
        else:
            mode = "local"
            print("Notice: AWS IoT certificates not found in .env. Defaulting to '--mode local'.")

    print(f"==================================================")
    print(f"🏭 Industrial Asset Telemetry Simulator")
    print(f"   Mode: {mode.upper()} | Fault: {fault.upper()} | Interval: {args.interval}s")
    print(f"==================================================")

    if mode == "local":
        try:
            while True:
                reading = build_reading(thing_name, asset_id, fault_mode=fault)
                publish_local(reading)
                if args.once:
                    break
                time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            print("\nStopped.")

    elif mode == "api":
        try:
            while True:
                reading = build_reading(thing_name, asset_id, fault_mode=fault)
                publish_api(reading, args.api_url)
                if args.once:
                    break
                time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            print("\nStopped.")

    elif mode == "mqtt":
        import paho.mqtt.client as mqtt

        endpoint = os.getenv("AWS_IOT_ENDPOINT")
        topic = os.getenv("AWS_IOT_TOPIC", f"factory/machines/{thing_name}/telemetry")
        cert = os.getenv("AWS_IOT_CERT")
        key = os.getenv("AWS_IOT_PRIVATE_KEY")
        root_ca = os.getenv("AWS_IOT_ROOT_CA")

        for label, path in [("certificate", cert), ("private key", key), ("Root CA", root_ca)]:
            if not path or not Path(path).is_file():
                raise SystemExit(f"{label} file not found: {path}. Use '--mode local' or '--mode api' instead.")

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=thing_name,
            protocol=mqtt.MQTTv311,
        )
        client.tls_set(ca_certs=root_ca, certfile=cert, keyfile=key)
        client.connect(endpoint, 8883, keepalive=60)
        client.loop_start()

        try:
            while True:
                reading = build_reading(thing_name, asset_id, fault_mode=fault)
                payload = json.dumps(reading)
                info = client.publish(topic, payload=payload, qos=1)
                info.wait_for_publish(timeout=10)
                print(f"[MQTT] Published to {topic}: Temp={reading['temperature']}°C | Vib={reading['vibration']}g")
                if args.once:
                    break
                time.sleep(max(1.0, args.interval))
        finally:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()
