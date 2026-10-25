"""
Serverless AWS Lambda Handlers for Edge-to-Cloud Telemetry Pipeline.
Matches Section 8.2 of the Capstone System Specification:
  1. telemetry_ingest_lambda: Validates, cleanses, enriches, and stores raw MQTT telemetry in S3 & Timestream
  2. feature_engineering_lambda: Extracts time-domain statistics and FFT spectral features
  3. inference_lambda: Executes unified PrognosticsEngine (Isolation Forest, AE, 1D-CNN, CNN-LSTM)
  4. alert_lambda: Evaluates risk thresholds, updates Digital Twin state, and triggers LangGraph Swarm
  5. sap_integration_lambda: Dispatches generated SAP PM Work Orders and SAP MM Purchase Requisitions
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import AssetConfig, DEFAULT_ASSET_CONFIG, FaultClass
from src.features.pipeline import FeatureExtractionPipeline
from src.models.engine import PrognosticsEngine, DiagnosticReport
from src.digital_twin.state_machine import DigitalTwin, AssetState
from src.agents.swarm import DecisionSwarm
from src.sap.client import SAPClient
from src.cloud.storage_adapter import CloudStorageAdapter


class ServerlessPipeline:
    """
    Simulated AWS Serverless Cloud Telemetry Pipeline.
    Encapsulates AWS Lambda functions and cloud storage adapters for IoT telemetry processing.
    """

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        storage_adapter: Optional[CloudStorageAdapter] = None,
        feature_pipeline: Optional[FeatureExtractionPipeline] = None,
        prognostics_engine: Optional[PrognosticsEngine] = None,
        digital_twin: Optional[DigitalTwin] = None,
        decision_swarm: Optional[DecisionSwarm] = None,
        sap_client: Optional[SAPClient] = None,
    ):
        self.config = config or DEFAULT_ASSET_CONFIG
        self.storage = storage_adapter or CloudStorageAdapter()
        self.feature_pipeline = feature_pipeline or FeatureExtractionPipeline(self.config)
        self.engine = prognostics_engine or PrognosticsEngine(config=self.config)
        self.twin = digital_twin or DigitalTwin(config=self.config)
        self.swarm = decision_swarm or DecisionSwarm()
        self.sap_client = sap_client or SAPClient()

    # -------------------------------------------------------------------------
    # Lambda 1: Telemetry Ingest
    # -------------------------------------------------------------------------
    def telemetry_ingest_lambda(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """
        Validates, cleanses, enriches, and archives incoming MQTT telemetry payload.
        Persists raw batch to S3 Data Lake and key metrics to Amazon Timestream.
        """
        # 1. Payload validation
        required_fields = ["asset_id", "vibration_x"]
        for f in required_fields:
            if f not in event:
                return {
                    "statusCode": 400,
                    "error": f"Missing required telemetry field: {f}",
                }

        asset_id = event["asset_id"]
        temp_c = float(event.get("temperature_c", 48.0))
        current_a = float(event.get("current_a", 1.20))
        rms_vib = float(event.get("rms_vibration", 0.25))

        # Sensor range validation / integrity check
        if temp_c < -40.0 or temp_c > 150.0:
            temp_c = 50.0  # Safe fallback
        if current_a < 0.0 or current_a > 10.0:
            current_a = 1.20

        # 2. Store in S3 raw data lake
        s3_uri = self.storage.store_raw_s3(asset_id, event)

        # 3. Store time-series metrics in Amazon Timestream
        self.storage.write_timestream_record(
            asset_id=asset_id,
            dimensions={"plant": event.get("plant", "PLANT-01"), "device_id": event.get("device_id", "ESP32-001")},
            measures={
                "rms_vibration": rms_vib,
                "temperature_c": temp_c,
                "current_a": current_a,
                "speed_rpm": float(event.get("speed_rpm", 1800.0)),
            },
        )

        return {
            "statusCode": 200,
            "ingest_status": "SUCCESS",
            "asset_id": asset_id,
            "s3_uri": s3_uri,
            "cleaned_telemetry": {
                "asset_id": asset_id,
                "temperature_c": temp_c,
                "current_a": current_a,
                "rms_vibration": rms_vib,
                "speed_rpm": float(event.get("speed_rpm", 1800.0)),
                "vibration_x": event["vibration_x"],
                "vibration_y": event.get("vibration_y"),
                "vibration_z": event.get("vibration_z"),
            },
        }

    # -------------------------------------------------------------------------
    # Lambda 2: Feature Engineering
    # -------------------------------------------------------------------------
    def feature_engineering_lambda(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """Extracts tabular time-domain and FFT spectral features from raw vibration window."""
        telem = event.get("cleaned_telemetry", event)
        accel_x = np.array(telem["vibration_x"], dtype=np.float64)
        accel_y = np.array(telem.get("vibration_y"), dtype=np.float64) if telem.get("vibration_y") else None
        accel_z = np.array(telem.get("vibration_z"), dtype=np.float64) if telem.get("vibration_z") else None

        n = len(accel_x)
        temp_arr = np.full(n, float(telem.get("temperature_c", 50.0)))
        curr_arr = np.full(n, float(telem.get("current_a", 1.20)))
        speed_arr = np.full(n, float(telem.get("speed_rpm", 1800.0)))

        features = self.feature_pipeline.extract_from_arrays(
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            temperature_c=temp_arr,
            current_a=curr_arr,
            speed_rpm=speed_arr,
        )

        return {
            "statusCode": 200,
            "asset_id": telem.get("asset_id", "MOTOR-001"),
            "features": features,
            "raw_arrays": {
                "accel_x": accel_x,
                "accel_y": accel_y,
                "accel_z": accel_z,
                "temperature_c": temp_arr,
                "current_a": curr_arr,
                "speed_rpm": speed_arr,
            },
        }

    # -------------------------------------------------------------------------
    # Lambda 3: Prognostics Inference
    # -------------------------------------------------------------------------
    def inference_lambda(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """Runs Multimodal Prognostics Engine (Isolation Forest, AE, 1D-CNN, CNN-LSTM)."""
        raw = event.get("raw_arrays")
        if not raw:
            return {"statusCode": 400, "error": "No raw arrays provided for inference"}

        report = self.engine.predict(
            accel_x=raw["accel_x"],
            accel_y=raw["accel_y"],
            accel_z=raw["accel_z"],
            temperature_c=raw["temperature_c"],
            current_a=raw["current_a"],
            speed_rpm=raw["speed_rpm"],
        )

        return {
            "statusCode": 200,
            "report": report,
            "diagnostic_dict": report.to_dict(),
        }

    # -------------------------------------------------------------------------
    # Lambda 4: Alert Evaluation & Decision Swarm Invocation
    # -------------------------------------------------------------------------
    def alert_lambda(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """
        Evaluates prognostic report, updates Digital Twin state machine,
        and triggers LangGraph Multi-Agent Decision Swarm if anomaly detected.
        """
        report: DiagnosticReport = event["report"]

        # Update Digital Twin State Machine & Section 9.3 Health Score
        state = self.twin.update_telemetry(report)

        # Trigger Decision Swarm
        swarm_result = self.swarm.run(report)

        # Store alert event in DynamoDB
        alert_record = {
            "asset_id": report.asset_id,
            "timestamp": report.timestamp,
            "state": state.value,
            "health_score": self.twin.health_score,
            "anomaly_score": report.anomaly_score,
            "failure_mode": report.predicted_failure_mode,
            "rul_hours": report.rul_hours,
            "swarm_status": swarm_result["swarm_status"],
        }
        self.storage.put_dynamodb_item(f"ALERT_{report.asset_id}_{report.timestamp}", alert_record)

        return {
            "statusCode": 200,
            "asset_state": state.value,
            "health_score": self.twin.health_score,
            "swarm_result": swarm_result,
            "is_action_required": swarm_result["swarm_status"] != "SKIPPED_HEALTHY",
        }

    # -------------------------------------------------------------------------
    # Lambda 5: SAP Enterprise Core Integration
    # -------------------------------------------------------------------------
    def sap_integration_lambda(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """Dispatches generated SAP PM Work Orders and SAP MM Purchase Requisitions to S/4HANA."""
        swarm_res = event.get("swarm_result", {})
        if not event.get("is_action_required", False) or swarm_res.get("swarm_status") == "SKIPPED_HEALTHY":
            return {
                "statusCode": 200,
                "message": "Asset operating nominally. No SAP work order dispatch required.",
                "dispatched": False,
            }

        notif_resp = None
        order_resp = None
        pr_resp = None

        # 1. Dispatch SAP PM Notification
        if swarm_res.get("sap_pm_notification"):
            notif_resp = self.sap_client.create_maintenance_notification(swarm_res["sap_pm_notification"])

        # 2. Dispatch SAP PM Work Order
        if swarm_res.get("sap_pm_order"):
            order_resp = self.sap_client.create_maintenance_order(swarm_res["sap_pm_order"])
            # Update Digital Twin to MAINTENANCE state
            self.twin.set_maintenance_active(order_resp["order_id"])

        # 3. Dispatch SAP MM Purchase Requisition if shortage occurred
        if swarm_res.get("sap_mm_pr"):
            pr_resp = self.sap_client.create_purchase_requisition(swarm_res["sap_mm_pr"])
            if pr_resp and pr_resp.get("purchase_requisition_id"):
                self.twin.active_purchase_requisitions.append(pr_resp["purchase_requisition_id"])

        return {
            "statusCode": 201,
            "dispatched": True,
            "notification_response": notif_resp,
            "order_response": order_resp,
            "pr_response": pr_resp,
            "digital_twin_state": self.twin.state.value,
            "active_work_order": self.twin.active_work_orders[0] if self.twin.active_work_orders else None,
        }

    # -------------------------------------------------------------------------
    # Complete Pipeline End-to-End Execution
    # -------------------------------------------------------------------------
    def execute_pipeline(self, raw_mqtt_message: Dict[str, Any]) -> Dict[str, Any]:
        """Chains all 5 serverless functions from edge ingest to closed-loop SAP dispatch."""
        step1 = self.telemetry_ingest_lambda(raw_mqtt_message)
        if step1.get("statusCode") != 200:
            return {"status": "FAILED", "step": "ingest", "error": step1}

        step2 = self.feature_engineering_lambda(step1)
        if step2.get("statusCode") != 200:
            return {"status": "FAILED", "step": "feature_engineering", "error": step2}

        step3 = self.inference_lambda(step2)
        if step3.get("statusCode") != 200:
            return {"status": "FAILED", "step": "inference", "error": step3}

        step4 = self.alert_lambda(step3)
        step5 = self.sap_integration_lambda(step4)

        return {
            "status": "COMPLETED",
            "ingest": step1,
            "inference": step3["diagnostic_dict"],
            "alert": {"asset_state": step4["asset_state"], "health_score": step4["health_score"]},
            "sap_dispatch": step5,
            "twin_overview": self.twin.to_dict(),
        }
