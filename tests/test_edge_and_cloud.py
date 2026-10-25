"""
Unit and Integration Tests for Step 7: Edge Sensing, MQTT Telemetry & Cloud Ingestion.
Validates:
  - ESP32 Firmware and Wokwi Simulation Circuit Configuration
  - Edge Telemetry Payload Framing and MQTT Client Buffering
  - AWS Cloud Storage Adapter (S3 Data Lake, Timestream, DynamoDB)
  - All 5 Serverless AWS Lambda Handlers (Section 8.2)
  - Full Closed-Loop Serverless Pipeline from Edge Ingestion to SAP Dispatch
"""

import json
import os
import pytest
import numpy as np

from src.config import AssetConfig, FaultClass
from src.digital_twin.simulator import PhysicalSignalSimulator
from src.edge.mqtt_client import EdgeTelemetryPayload, EdgeMQTTClient
from src.cloud.storage_adapter import CloudStorageAdapter
from src.cloud.lambda_handlers import ServerlessPipeline
from src.digital_twin.state_machine import AssetState


@pytest.fixture
def simulator():
    return PhysicalSignalSimulator()


@pytest.fixture
def storage_adapter():
    return CloudStorageAdapter()


@pytest.fixture
def serverless_pipeline(storage_adapter):
    return ServerlessPipeline(storage_adapter=storage_adapter)


# -----------------------------------------------------------------------------
# 1. Edge Firmware & Wokwi Simulation Verification
# -----------------------------------------------------------------------------
def test_wokwi_circuit_configuration():
    """Verify Wokwi simulation diagram.json and wokwi.toml circuit definitions."""
    diagram_path = "src/edge/firmware/diagram.json"
    toml_path = "src/edge/firmware/wokwi.toml"

    assert os.path.exists(diagram_path)
    assert os.path.exists(toml_path)

    with open(diagram_path, "r", encoding="utf-8") as f:
        diagram = json.load(f)

    assert "parts" in diagram
    assert "connections" in diagram

    part_types = [p["type"] for p in diagram["parts"]]
    assert "board-esp32-devkit-c-v4" in part_types
    assert "wokwi-mpu6050" in part_types
    assert "wokwi-ssd1306" in part_types


def test_firmware_file_specifications():
    """Verify ESP32 firmware source code has required sensors, topics, and failover logic."""
    firmware_path = "src/edge/firmware/firmware.ino"
    assert os.path.exists(firmware_path)

    with open(firmware_path, "r", encoding="utf-8") as f:
        code = f.read()

    assert "factory/plant01/motor001/telemetry" in code
    assert "MPU6050_ADDR" in code
    assert "DallasTemperature" in code
    assert "ACS712" in code
    assert "Adafruit_SSD1306" in code
    assert "bufferToSD" in code


# -----------------------------------------------------------------------------
# 2. Edge MQTT Telemetry Client Tests
# -----------------------------------------------------------------------------
def test_edge_telemetry_payload_serialization(simulator):
    """Test standard JSON serialization of edge telemetry bursts."""
    batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=0.5, seed=12)
    payload = EdgeTelemetryPayload.from_batch(batch)

    json_str = payload.to_json()
    data = json.loads(json_str)

    assert data["asset_id"] == "MOTOR-001"
    assert data["device_id"] == "ESP32-MOTOR-001"
    assert data["sampling_rate_hz"] == 1000.0
    assert len(data["vibration_x"]) == len(batch.accel_x)
    assert data["rms_vibration"] > 0.0


def test_edge_mqtt_client_publish_and_offline_buffering(simulator):
    """Test MQTT publishing and local buffering on connection drop."""
    client = EdgeMQTTClient(simulate=True)
    batch = simulator.generate_window(FaultClass.HEALTHY, duration_sec=0.2, seed=1)
    payload = EdgeTelemetryPayload.from_batch(batch)

    # 1. When disconnected, payload must buffer locally
    assert client.is_connected is False
    published = client.publish_telemetry(payload)
    assert published is False
    assert len(client.offline_buffer) == 1

    # 2. When connected, buffer must flush
    client.connect()
    assert client.is_connected is True
    assert len(client.offline_buffer) == 0
    assert len(client.published_messages) == 1
    assert client.published_messages[0]["topic"] == EdgeMQTTClient.TOPIC_TELEMETRY

    # 3. Publish alert
    alert_ok = client.publish_alert({"alert": "High vibration detected", "severity": "HIGH"})
    assert alert_ok is True
    assert client.published_messages[1]["topic"] == EdgeMQTTClient.TOPIC_ALERT


# -----------------------------------------------------------------------------
# 3. Cloud Storage Adapter Tests (S3, Timestream, DynamoDB)
# -----------------------------------------------------------------------------
def test_cloud_storage_adapter(storage_adapter):
    """Test raw telemetry archiving in S3, Timestream writing, and DynamoDB item operations."""
    asset_id = "MOTOR-001"
    sample_data = {"rms": 0.28, "temp": 50.0}

    # S3 Store
    s3_uri = storage_adapter.store_raw_s3(asset_id, sample_data)
    assert s3_uri.startswith("s3://industrial-iot-telemetry-lake/raw/")
    assert len(storage_adapter.s3_objects) == 1

    # Timestream Write
    records_written = storage_adapter.write_timestream_record(
        asset_id=asset_id,
        dimensions={"plant": "PLANT-01"},
        measures={"rms_vibration": 0.28, "temperature_c": 50.0},
    )
    assert records_written == 2
    assert len(storage_adapter.timestream_records) == 1

    # DynamoDB Store & Retrieve
    storage_adapter.put_dynamodb_item("ASSET_MOTOR-001", {"status": "RUNNING", "plant": "PLANT-01"})
    item = storage_adapter.get_dynamodb_item("ASSET_MOTOR-001")
    assert item is not None
    assert item["status"] == "RUNNING"


# -----------------------------------------------------------------------------
# 4. Serverless Lambda Handlers Tests (Section 8.2)
# -----------------------------------------------------------------------------
def test_serverless_telemetry_ingest_lambda(serverless_pipeline):
    """Test telemetry_ingest_lambda validation, S3 storage, and Timestream metric writes."""
    # Valid payload
    event = {
        "asset_id": "MOTOR-001",
        "device_id": "ESP32-MOTOR-001",
        "plant": "PLANT-01",
        "temperature_c": 49.5,
        "current_a": 1.22,
        "rms_vibration": 0.24,
        "speed_rpm": 1800.0,
        "vibration_x": [0.1, -0.1, 0.2, -0.2],
    }
    resp = serverless_pipeline.telemetry_ingest_lambda(event)
    assert resp["statusCode"] == 200
    assert resp["ingest_status"] == "SUCCESS"
    assert "s3_uri" in resp
    assert resp["cleaned_telemetry"]["asset_id"] == "MOTOR-001"

    # Missing mandatory field
    bad_event = {"temperature_c": 49.5}
    bad_resp = serverless_pipeline.telemetry_ingest_lambda(bad_event)
    assert bad_resp["statusCode"] == 400


def test_serverless_feature_engineering_lambda(serverless_pipeline, simulator):
    """Test feature_engineering_lambda extracting tabular and spectral features from raw array."""
    batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=0.5, seed=5)
    payload = EdgeTelemetryPayload.from_batch(batch)
    event = json.loads(payload.to_json())

    ingest_res = serverless_pipeline.telemetry_ingest_lambda(event)
    feat_res = serverless_pipeline.feature_engineering_lambda(ingest_res)

    assert feat_res["statusCode"] == 200
    features = feat_res["features"]
    assert "rad_rms" in features
    assert "rad_kurtosis" in features
    assert "rad_dominant_freq_hz" in features


def test_serverless_inference_and_alert_lambdas(serverless_pipeline, simulator):
    """Test inference_lambda prognostics evaluation and alert_lambda Swarm routing."""
    batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=0.5, seed=15)
    payload = EdgeTelemetryPayload.from_batch(batch)
    event = json.loads(payload.to_json())

    ingest_res = serverless_pipeline.telemetry_ingest_lambda(event)
    feat_res = serverless_pipeline.feature_engineering_lambda(ingest_res)
    infer_res = serverless_pipeline.inference_lambda(feat_res)

    assert infer_res["statusCode"] == 200
    assert infer_res["report"].is_anomaly is True
    assert infer_res["report"].predicted_class == FaultClass.BEARING_WEAR

    alert_res = serverless_pipeline.alert_lambda(infer_res)
    assert alert_res["statusCode"] == 200
    assert alert_res["asset_state"] in (AssetState.WARNING.value, AssetState.CRITICAL.value)
    assert alert_res["is_action_required"] is True


def test_serverless_sap_integration_lambda(serverless_pipeline, simulator):
    """Test sap_integration_lambda dispatching Work Order and updating Digital Twin."""
    batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=0.5, seed=22)
    payload = EdgeTelemetryPayload.from_batch(batch)
    event = json.loads(payload.to_json())

    ingest_res = serverless_pipeline.telemetry_ingest_lambda(event)
    feat_res = serverless_pipeline.feature_engineering_lambda(ingest_res)
    infer_res = serverless_pipeline.inference_lambda(feat_res)
    alert_res = serverless_pipeline.alert_lambda(infer_res)
    sap_res = serverless_pipeline.sap_integration_lambda(alert_res)

    assert sap_res["statusCode"] == 201
    assert sap_res["dispatched"] is True
    assert sap_res["order_response"]["success"] is True
    assert sap_res["digital_twin_state"] == AssetState.MAINTENANCE.value


# -----------------------------------------------------------------------------
# 5. Full End-to-End Serverless Pipeline Execution
# -----------------------------------------------------------------------------
def test_full_pipeline_execute_from_raw_edge_message(serverless_pipeline, simulator):
    """
    Complete closed-loop verification:
    Raw Edge MQTT JSON Message -> AWS IoT Lambda Pipeline -> S3/Timestream Storage ->
    Feature Extraction -> Multimodal Prognostics -> LangGraph Swarm ->
    SAP S/4HANA OData Dispatch -> Digital Twin Maintenance State!
    """
    batch = simulator.generate_window(FaultClass.BEARING_WEAR, duration_sec=1.0, seed=77)
    payload = EdgeTelemetryPayload.from_batch(batch)
    raw_mqtt_msg = json.loads(payload.to_json())

    result = serverless_pipeline.execute_pipeline(raw_mqtt_msg)

    assert result["status"] == "COMPLETED"
    assert result["ingest"]["statusCode"] == 200
    assert result["inference"]["predicted_failure_mode"] == "Bearing Wear (BPFO)"
    assert result["sap_dispatch"]["dispatched"] is True
    assert result["twin_overview"]["state"] == "MAINTENANCE"
    assert result["twin_overview"]["active_work_order"] is not None
