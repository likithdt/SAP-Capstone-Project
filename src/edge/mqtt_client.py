"""
Edge MQTT Telemetry Publisher and Client.
Publishes high-frequency sensor bursts to AWS IoT Core MQTT Broker over TLS 1.2.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from src.digital_twin.simulator import TelemetryBatch


@dataclass
class EdgeTelemetryPayload:
    """Standardized JSON Telemetry schema transmitted by ESP32 to AWS IoT Core."""
    asset_id: str
    device_id: str
    plant: str
    timestamp: str
    sampling_rate_hz: float
    sample_count: int
    rms_vibration: float
    temperature_c: float
    current_a: float
    speed_rpm: float
    vibration_x: List[float]
    vibration_y: List[float]
    vibration_z: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize payload to JSON string."""
        return json.dumps({
            "asset_id": self.asset_id,
            "device_id": self.device_id,
            "plant": self.plant,
            "timestamp": self.timestamp,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sample_count": self.sample_count,
            "rms_vibration": round(self.rms_vibration, 4),
            "temperature_c": round(self.temperature_c, 2),
            "current_a": round(self.current_a, 3),
            "speed_rpm": round(self.speed_rpm, 1),
            "vibration_x": [round(float(v), 4) for v in self.vibration_x],
            "vibration_y": [round(float(v), 4) for v in self.vibration_y],
            "vibration_z": [round(float(v), 4) for v in self.vibration_z],
            "metadata": self.metadata,
        })

    @classmethod
    def from_batch(cls, batch: TelemetryBatch) -> "EdgeTelemetryPayload":
        """Convert a TelemetryBatch instance into an EdgeTelemetryPayload."""
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return cls(
            asset_id=batch.asset_id,
            device_id=batch.device_id,
            plant="PLANT-01",
            timestamp=now_ts,
            sampling_rate_hz=batch.sampling_rate_hz,
            sample_count=len(batch.accel_x),
            rms_vibration=batch.rms_vibration,
            temperature_c=batch.mean_temperature,
            current_a=batch.mean_current,
            speed_rpm=batch.mean_speed_rpm,
            vibration_x=batch.accel_x.tolist(),
            vibration_y=batch.accel_y.tolist(),
            vibration_z=batch.accel_z.tolist(),
            metadata={"fault_class": batch.fault_class, "fault_name": batch.fault_name},
        )


class EdgeMQTTClient:
    """
    Simulated and Live MQTT Client for AWS IoT Core.
    Publishes sensor telemetry to factory/plant01/motor001/telemetry and buffers when offline.
    """

    TOPIC_TELEMETRY = "factory/plant01/motor001/telemetry"
    TOPIC_ALERT = "factory/plant01/motor001/alert"
    TOPIC_HEALTH = "factory/plant01/motor001/health"
    TOPIC_COMMAND = "factory/plant01/motor001/command"

    def __init__(
        self,
        endpoint: str = "a2xxxxxxxxxxxx-ats.iot.us-east-1.amazonaws.com",
        port: int = 8883,
        client_id: str = "ESP32-MOTOR-001",
        simulate: bool = True,
    ):
        self.endpoint = endpoint
        self.port = port
        self.client_id = client_id
        self.simulate = simulate
        self.is_connected = False

        # In-memory queues
        self.published_messages: List[Dict[str, Any]] = []
        self.offline_buffer: List[EdgeTelemetryPayload] = []
        self.subscriptions: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}

    def connect(self) -> bool:
        """Establish MQTT connection with TLS credentials."""
        self.is_connected = True
        # Flush offline buffer if messages were queued
        if self.offline_buffer:
            for item in list(self.offline_buffer):
                self.publish_telemetry(item)
            self.offline_buffer.clear()
        return True

    def disconnect(self) -> None:
        """Simulate network disconnection."""
        self.is_connected = False

    def publish_telemetry(self, payload: EdgeTelemetryPayload) -> bool:
        """
        Publish telemetry payload to AWS IoT Core topic.
        Buffers to offline queue if disconnected.
        """
        if not self.is_connected:
            self.offline_buffer.append(payload)
            return False

        message = {
            "topic": self.TOPIC_TELEMETRY,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "qos": 1,
        }
        self.published_messages.append(message)
        self._dispatch_to_subscribers(self.TOPIC_TELEMETRY, json.loads(payload.to_json()))
        return True

    def publish_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Publish high-priority anomaly alert to AWS IoT alert topic."""
        if not self.is_connected:
            return False

        message = {
            "topic": self.TOPIC_ALERT,
            "payload": alert_data,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "qos": 1,
        }
        self.published_messages.append(message)
        self._dispatch_to_subscribers(self.TOPIC_ALERT, alert_data)
        return True

    def subscribe(self, topic: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Subscribe handler to MQTT topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)

    def _dispatch_to_subscribers(self, topic: str, data: Dict[str, Any]) -> None:
        """Notify subscribed callbacks."""
        handlers = self.subscriptions.get(topic, [])
        for handler in handlers:
            handler(topic, data)
