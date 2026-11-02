import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

def build_reading(device_id: str, abnormal: bool = False) -> dict:
    temperature = random.uniform(45, 78)
    vibration = random.uniform(1, 7)
    humidity = random.uniform(35, 70)
    motor_status = "RUNNING"

    if abnormal:
        # Simulated fault conditions for presentation; not real safety thresholds.
        temperature = random.uniform(95, 115)
        vibration = random.uniform(12, 20)

    return {
        "deviceId": device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "temperature": round(temperature, 2),
        "vibration": round(vibration, 2),
        "humidity": round(humidity, 2),
        "motorStatus": motor_status,
    }

def main():
    parser = argparse.ArgumentParser(description="Publish simulated industrial telemetry to AWS IoT Core.")
    parser.add_argument("--once", action="store_true", help="publish one message and exit")
    parser.add_argument("--abnormal", action="store_true", help="publish simulated high temperature/vibration")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between messages (default: 5)")
    args = parser.parse_args()

    endpoint = os.getenv("AWS_IOT_ENDPOINT")
    thing_name = os.getenv("AWS_IOT_THING_NAME", "industrial-machine-01")
    topic = os.getenv("AWS_IOT_TOPIC", f"factory/machines/{thing_name}/telemetry")
    cert = os.getenv("AWS_IOT_CERT")
    key = os.getenv("AWS_IOT_PRIVATE_KEY")
    root_ca = os.getenv("AWS_IOT_ROOT_CA")

    missing = [name for name, value in {
        "AWS_IOT_ENDPOINT": endpoint,
        "AWS_IOT_CERT": cert,
        "AWS_IOT_PRIVATE_KEY": key,
        "AWS_IOT_ROOT_CA": root_ca,
    }.items() if not value]
    if missing:
        raise SystemExit("Missing settings in .env: " + ", ".join(missing))

    for label, path in [("certificate", cert), ("private key", key), ("Root CA", root_ca)]:
        if not Path(path).is_file():
            raise SystemExit(f"{label} file not found: {path}")

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
            reading = build_reading(thing_name, abnormal=args.abnormal)
            payload = json.dumps(reading)
            info = client.publish(topic, payload=payload, qos=1)
            info.wait_for_publish(timeout=10)
            if not info.is_published():
                raise RuntimeError("MQTT publish did not complete")
            print(f"Published to {topic}:\n{json.dumps(reading, indent=2)}\n")
            if args.once:
                break
            time.sleep(max(args.interval, 1.0))
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
