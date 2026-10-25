"""
Digital Twin State Machine and Composite Health Score Engine.
Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine with Closed-Loop ERP Action.
Matches Sections 9.1, 9.2, and 9.3 of the Capstone System Specification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import AssetConfig, DEFAULT_ASSET_CONFIG, FaultClass, ClassThresholds
from src.models.engine import DiagnosticReport


class AssetState(str, Enum):
    """
    Standard Asset Operational States matching Section 9.2:
      - HEALTHY: Normal vibration, temperature, and current behavior
      - WATCHLIST: Minor deviation; monitoring frequency increases
      - WARNING: Anomaly detected; maintenance should be planned
      - CRITICAL: High failure probability; urgent maintenance required
      - MAINTENANCE: SAP PM work order created; asset under active maintenance
      - OFFLINE: No telemetry received / sensor disconnection
    """
    HEALTHY = "HEALTHY"
    WATCHLIST = "WATCHLIST"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"

    @property
    def color_code(self) -> str:
        """Hex color representation for SAP Fiori dashboard rendering."""
        colors = {
            AssetState.HEALTHY: "#2B7D2B",     # Green
            AssetState.WATCHLIST: "#E6B800",   # Yellow
            AssetState.WARNING: "#E66000",     # Orange
            AssetState.CRITICAL: "#BB0000",    # Red
            AssetState.MAINTENANCE: "#107E3E", # SAP Blue/Green
            AssetState.OFFLINE: "#666666",     # Gray
        }
        return colors[self]


@dataclass
class HealthScoreWeights:
    """Configurable weights for Section 9.3 composite health score formula."""
    w1_anomaly: float = 40.0       # Weight for normalized composite anomaly score
    w2_temperature: float = 25.0   # Weight for thermal risk above nominal 52 °C
    w3_current: float = 20.0       # Weight for current overload risk above 1.30 A
    w4_rul: float = 15.0           # Weight for Remaining Useful Life degradation

    def __post_init__(self):
        total = self.w1_anomaly + self.w2_temperature + self.w3_current + self.w4_rul
        if abs(total - 100.0) > 1e-3:
            raise ValueError(f"Health score weights must sum to 100.0 (got {total:.2f})")


@dataclass
class HealthScoreBreakdown:
    """Detailed breakdown of composite health score components."""
    health_score: float            # Scaled 0.0 - 100.0
    anomaly_penalty: float         # w1 * A
    temperature_penalty: float     # w2 * T
    current_penalty: float         # w3 * C
    rul_penalty: float             # w4 * R
    normalized_anomaly: float      # A in [0, 1]
    normalized_temperature: float  # T in [0, 1]
    normalized_current: float      # C in [0, 1]
    normalized_rul_risk: float     # R in [0, 1]


class HealthScoreEngine:
    """
    Computes Section 9.3 composite asset health score:
      HealthScore = 100 - (w1 * A + w2 * T + w3 * C + w4 * R)
    """

    def __init__(self, weights: Optional[HealthScoreWeights] = None):
        self.weights = weights or HealthScoreWeights()

    def compute(
        self,
        anomaly_score: float,
        temperature_c: float,
        current_a: float,
        rul_hours: float,
        max_rul_hours: float = 120.0,
    ) -> HealthScoreBreakdown:
        """
        Calculate composite health score and penalization breakdown.

        Args:
            anomaly_score: Composite anomaly score in [0.0, 1.0]
            temperature_c: Mean or maximum motor temperature (°C)
            current_a: Mean or peak motor current draw (A)
            rul_hours: Remaining Useful Life estimate in hours
            max_rul_hours: Reference nominal healthy RUL window (120 hours)
        """
        # 1. Normalized Anomaly Score (A)
        A = float(np.clip(anomaly_score, 0.0, 1.0))

        # 2. Normalized Temperature Risk (T): 0 at <= 52.0 °C, 1.0 at >= 85.0 °C
        t_min = ClassThresholds.HEALTHY_TEMP_MAX_C
        t_max = ClassThresholds.THERMAL_OVERHEATING_CRITICAL_C
        T = float(np.clip((temperature_c - t_min) / (t_max - t_min), 0.0, 1.0))

        # 3. Normalized Current Overload Risk (C): 0 at <= 1.30 A, 1.0 at >= 2.40 A
        c_min = ClassThresholds.HEALTHY_CURRENT_MAX_A
        c_max = ClassThresholds.OVERLOAD_CURRENT_MIN_A
        C = float(np.clip((current_a - c_min) / (c_max - c_min), 0.0, 1.0))

        # 4. Normalized RUL Risk (R): 0 when RUL = max_rul, 1.0 when RUL = 0
        R = float(np.clip((max_rul_hours - rul_hours) / max_rul_hours, 0.0, 1.0))

        # 5. Composite Penalties
        pen_a = self.weights.w1_anomaly * A
        pen_t = self.weights.w2_temperature * T
        pen_c = self.weights.w3_current * C
        pen_r = self.weights.w4_rul * R

        total_penalty = pen_a + pen_t + pen_c + pen_r
        score = float(np.clip(100.0 - total_penalty, 0.0, 100.0))

        return HealthScoreBreakdown(
            health_score=round(score, 1),
            anomaly_penalty=round(pen_a, 2),
            temperature_penalty=round(pen_t, 2),
            current_penalty=round(pen_c, 2),
            rul_penalty=round(pen_r, 2),
            normalized_anomaly=round(A, 4),
            normalized_temperature=round(T, 4),
            normalized_current=round(C, 4),
            normalized_rul_risk=round(R, 4),
        )


@dataclass
class DigitalTwinStateHistory:
    """Historical record of asset state transitions."""
    timestamp: str
    previous_state: AssetState
    new_state: AssetState
    health_score: float
    trigger_reason: str


class DigitalTwin:
    """
    Virtual Asset Representation & State Machine Engine for Rotating Industrial Asset.
    Maintains virtual model state, processes live telemetry, evaluates health score,
    and governs transitions across operational states.
    """

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        health_engine: Optional[HealthScoreEngine] = None,
        sap_equipment_number: str = "10000021",
    ):
        self.config = config or DEFAULT_ASSET_CONFIG
        self.health_engine = health_engine or HealthScoreEngine()
        self.sap_equipment_number = sap_equipment_number

        # Core State Variables
        self.asset_id: str = self.config.asset_id
        self.asset_name: str = self.config.asset_name
        self.plant: str = self.config.plant
        self.criticality: str = "HIGH"
        self.current_status: str = "RUNNING"
        self.state: AssetState = AssetState.HEALTHY
        self.health_score: float = 100.0
        self.anomaly_score: float = 0.0
        self.risk_level: str = "LOW"
        self.predicted_failure_mode: str = "Healthy"
        self.rul_hours: float = 120.0
        self.temperature_c: float = 48.5
        self.current_a: float = 1.20
        self.rms_vibration: float = 0.22
        self.speed_rpm: float = 1800.0

        # ERP Linkages
        self.active_work_orders: List[str] = []
        self.active_purchase_requisitions: List[str] = []
        self.last_diagnostic_report: Optional[DiagnosticReport] = None
        self.last_health_breakdown: Optional[HealthScoreBreakdown] = None

        # History and Audit Trace
        self.state_history: List[DigitalTwinStateHistory] = []
        self.telemetry_history: List[Dict[str, Any]] = []

    def update_telemetry(
        self,
        report: DiagnosticReport,
        rms_vibration: Optional[float] = None,
        temperature_c: Optional[float] = None,
        current_a: Optional[float] = None,
        speed_rpm: Optional[float] = None,
    ) -> AssetState:
        """
        Update Digital Twin with latest prognostics inference report and sensor observations.
        Executes Section 9.3 health score calculation and state machine transition logic.
        """
        self.last_diagnostic_report = report
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Update telemetry values from features or explicit overrides
        feats = report.features or {}
        self.rms_vibration = rms_vibration or float(feats.get("rad_rms", self.rms_vibration))
        self.temperature_c = temperature_c or float(feats.get("temp_mean", feats.get("temp_max", self.temperature_c)))
        self.current_a = current_a or float(feats.get("current_mean", self.current_a))
        self.speed_rpm = speed_rpm or float(feats.get("speed_mean", self.speed_rpm))

        self.anomaly_score = report.anomaly_score
        self.risk_level = report.risk_level
        self.predicted_failure_mode = report.predicted_failure_mode
        self.rul_hours = report.rul_hours

        # Compute Section 9.3 Composite Health Score
        breakdown = self.health_engine.compute(
            anomaly_score=self.anomaly_score,
            temperature_c=self.temperature_c,
            current_a=self.current_a,
            rul_hours=self.rul_hours,
        )
        self.last_health_breakdown = breakdown
        self.health_score = breakdown.health_score

        # Evaluate State Machine Transition
        new_state = self._evaluate_state_transition()
        if new_state != self.state:
            self._transition_to(new_state, reason=f"Telemetry update: HealthScore={self.health_score}, Fault={self.predicted_failure_mode}")

        # Record history
        self.telemetry_history.append({
            "timestamp": now_ts,
            "health_score": self.health_score,
            "anomaly_score": self.anomaly_score,
            "state": self.state.value,
            "rms_vibration": self.rms_vibration,
            "temperature_c": self.temperature_c,
            "current_a": self.current_a,
            "rul_hours": self.rul_hours,
        })
        # Keep sliding history capped to 500 samples
        if len(self.telemetry_history) > 500:
            self.telemetry_history.pop(0)

        return self.state

    def _evaluate_state_transition(self) -> AssetState:
        """State transition rules based on health score, anomaly severity, and RUL."""
        # If currently under maintenance, stay in MAINTENANCE until cleared
        if self.state == AssetState.MAINTENANCE and len(self.active_work_orders) > 0:
            return AssetState.MAINTENANCE

        # Critical: HealthScore < 45.0, Anomaly Score >= 0.85, RUL < 12h, or Severe Overload
        if self.health_score < 45.0 or self.anomaly_score >= 0.85 or self.rul_hours < 12.0:
            return AssetState.CRITICAL

        # Warning: HealthScore 45.0 - 65.0, or confirmed non-healthy fault class
        if self.health_score < 65.0 or (self.last_diagnostic_report and self.last_diagnostic_report.predicted_class != 0):
            return AssetState.WARNING

        # Watchlist: HealthScore 65.0 - 80.0, slight deviation from baseline
        if self.health_score < 80.0 or self.anomaly_score > 0.25:
            return AssetState.WATCHLIST

        return AssetState.HEALTHY

    def _transition_to(self, new_state: AssetState, reason: str) -> None:
        """Execute state transition and append to state audit history."""
        prev = self.state
        self.state = new_state
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.state_history.append(DigitalTwinStateHistory(
            timestamp=now_ts,
            previous_state=prev,
            new_state=new_state,
            health_score=self.health_score,
            trigger_reason=reason,
        ))

    def set_maintenance_active(self, order_id: str) -> None:
        """Mark asset in active maintenance upon SAP PM Work Order generation."""
        if order_id not in self.active_work_orders:
            self.active_work_orders.append(order_id)
        self.current_status = "MAINTENANCE"
        self._transition_to(
            AssetState.MAINTENANCE,
            reason=f"Active SAP PM Maintenance Order created: {order_id}",
        )

    def complete_maintenance(self, order_id: str) -> None:
        """Clear completed work order and return asset to service."""
        if order_id in self.active_work_orders:
            self.active_work_orders.remove(order_id)
        if len(self.active_work_orders) == 0:
            self.current_status = "RUNNING"
            self.state = AssetState.HEALTHY
            self.health_score = 98.0
            self.anomaly_score = 0.05
            self._transition_to(
                AssetState.HEALTHY,
                reason=f"Maintenance order {order_id} closed. Return to nominal operation.",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert virtual model state to Section 9.1 standardized JSON representation."""
        return {
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "plant": self.plant,
            "criticality": self.criticality,
            "current_status": self.current_status,
            "state": self.state.value,
            "health_score": self.health_score,
            "anomaly_score": round(self.anomaly_score, 4),
            "risk_level": self.risk_level,
            "predicted_failure_mode": self.predicted_failure_mode,
            "rul_hours": round(self.rul_hours, 1),
            "temperature_c": round(self.temperature_c, 1),
            "current_a": round(self.current_a, 2),
            "rms_vibration": round(self.rms_vibration, 3),
            "speed_rpm": round(self.speed_rpm, 1),
            "sap_equipment_number": self.sap_equipment_number,
            "active_work_order": self.active_work_orders[0] if self.active_work_orders else None,
            "active_work_orders": list(self.active_work_orders),
            "active_purchase_requisitions": list(self.active_purchase_requisitions),
        }
