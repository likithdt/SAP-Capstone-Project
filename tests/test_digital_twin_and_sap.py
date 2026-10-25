"""
Unit and Integration Tests for Step 6: SAP Enterprise Core & Digital Twin Fiori UI.
Validates:
  - Section 9.3 Composite Health Score Engine: HealthScore = 100 - (w1*A + w2*T + w3*C + w4*R)
  - Section 9.2 Asset State Machine Transitions: Healthy -> Watchlist -> Warning -> Critical -> Maintenance
  - SAP S/4HANA OData v4 Client for PM Work Orders and MM Purchase Requisitions
  - SAP Fiori Overview Page (OVP) Analytical and Operational Dashboard Cards
  - End-to-End Closed Loop: Physical Telemetry -> Prognostics -> Swarm -> SAP Dispatch -> Digital Twin Maintenance
"""

import pytest
from src.config import AssetConfig, FaultClass, SAPConfig
from src.digital_twin.simulator import PhysicalSignalSimulator
from src.digital_twin.state_machine import (
    AssetState,
    HealthScoreWeights,
    HealthScoreEngine,
    DigitalTwin,
)
from src.digital_twin.fiori_adapter import FioriDashboardAdapter
from src.models.engine import PrognosticsEngine, DiagnosticReport
from src.agents.swarm import DecisionSwarm
from src.sap.client import SAPClient


@pytest.fixture
def health_engine():
    return HealthScoreEngine()


@pytest.fixture
def digital_twin(health_engine):
    return DigitalTwin(health_engine=health_engine)


@pytest.fixture
def sap_client():
    return SAPClient()


# -----------------------------------------------------------------------------
# 1. Health Score Engine Tests (Section 9.3)
# -----------------------------------------------------------------------------
def test_health_score_engine_nominal_conditions(health_engine):
    """Test health score under nominal healthy conditions (> 90)."""
    res = health_engine.compute(
        anomaly_score=0.08,
        temperature_c=48.0,
        current_a=1.20,
        rul_hours=120.0,
    )
    assert res.health_score >= 95.0
    assert res.temperature_penalty == 0.0  # Under nominal 52.0 °C
    assert res.current_penalty == 0.0      # Under nominal 1.30 A
    assert res.rul_penalty == 0.0          # Max RUL remaining


def test_health_score_engine_bearing_wear_degradation(health_engine):
    """Test health score drop during developing bearing defect."""
    res = health_engine.compute(
        anomaly_score=0.82,
        temperature_c=62.0,
        current_a=1.35,
        rul_hours=36.0,
    )
    # Anomaly penalty w1*0.82 ~ 32.8
    # Temperature penalty w2*((62-52)/33) ~ 7.5
    # Current penalty w3*((1.35-1.30)/1.10) ~ 0.9
    # RUL penalty w4*((120-36)/120) ~ 10.5
    # Score ~ 100 - 51.7 ~ 48.3
    assert 40.0 <= res.health_score <= 60.0
    assert res.anomaly_penalty > 30.0


def test_health_score_engine_severe_overload_and_runaway(health_engine):
    """Test severe failure state pushes health score into critical territory (< 30)."""
    res = health_engine.compute(
        anomaly_score=0.98,
        temperature_c=86.5,
        current_a=2.85,
        rul_hours=4.0,
    )
    assert res.health_score < 25.0
    assert res.normalized_temperature == 1.0  # Above 85.0 °C max
    assert res.normalized_current == 1.0      # Above 2.40 A max


def test_health_score_weights_validation():
    """Verify that invalid weights summing != 100.0 raise ValueError."""
    with pytest.raises(ValueError):
        HealthScoreWeights(w1_anomaly=50.0, w2_temperature=50.0, w3_current=20.0, w4_rul=10.0)


# -----------------------------------------------------------------------------
# 2. Digital Twin State Machine Tests (Section 9.2)
# -----------------------------------------------------------------------------
def test_digital_twin_state_transitions(digital_twin):
    """Test full progression: Healthy -> Watchlist -> Warning -> Critical -> Maintenance -> Healthy."""
    assert digital_twin.state == AssetState.HEALTHY

    # 1. Minor drift -> WATCHLIST
    r_watch = DiagnosticReport(
        asset_id="MOTOR-001",
        timestamp="2026-10-25T14:00:00Z",
        is_anomaly=False,
        anomaly_score=0.28,
        risk_level="MEDIUM",
        predicted_class=0,
        predicted_failure_mode="Healthy",
        confidence=0.85,
        rul_hours=105.0,
        reconstruction_error=0.015,
        isolation_forest_score=0.30,
        evidence=["Slight vibration drift"],
        recommended_action="Continue monitoring",
        features={"rad_rms": 0.32, "temp_mean": 51.0, "current_mean": 1.25},
    )
    digital_twin.update_telemetry(r_watch)
    assert digital_twin.state == AssetState.WATCHLIST

    # 2. Confirmed Fault Class 1 -> WARNING
    r_warn = DiagnosticReport(
        asset_id="MOTOR-001",
        timestamp="2026-10-25T14:10:00Z",
        is_anomaly=True,
        anomaly_score=0.72,
        risk_level="HIGH",
        predicted_class=FaultClass.BEARING_WEAR,
        predicted_failure_mode="Bearing Wear (BPFO)",
        confidence=0.88,
        rul_hours=38.0,
        reconstruction_error=0.032,
        isolation_forest_score=0.70,
        evidence=["Elevated kurtosis"],
        recommended_action="Inspect bearing",
        features={"rad_rms": 0.65, "temp_mean": 58.0, "current_mean": 1.30},
    )
    digital_twin.update_telemetry(r_warn)
    assert digital_twin.state == AssetState.WARNING

    # 3. Severe Overheating -> CRITICAL
    r_crit = DiagnosticReport(
        asset_id="MOTOR-001",
        timestamp="2026-10-25T14:20:00Z",
        is_anomaly=True,
        anomaly_score=0.92,
        risk_level="CRITICAL",
        predicted_class=FaultClass.THERMAL_OVERHEATING,
        predicted_failure_mode="Thermal Overheating",
        confidence=0.95,
        rul_hours=8.0,
        reconstruction_error=0.055,
        isolation_forest_score=0.90,
        evidence=["Stator temperature 86 °C"],
        recommended_action="Emergency shutdown",
        features={"rad_rms": 0.80, "temp_mean": 86.0, "current_mean": 2.20},
    )
    digital_twin.update_telemetry(r_crit)
    assert digital_twin.state == AssetState.CRITICAL

    # 4. SAP Work Order Created -> MAINTENANCE
    digital_twin.set_maintenance_active("40003001")
    assert digital_twin.state == AssetState.MAINTENANCE
    assert "40003001" in digital_twin.active_work_orders

    # 5. Maintenance Completed -> HEALTHY
    digital_twin.complete_maintenance("40003001")
    assert digital_twin.state == AssetState.HEALTHY
    assert len(digital_twin.active_work_orders) == 0


# -----------------------------------------------------------------------------
# 3. SAP Client Tests (PM & MM OData Integration)
# -----------------------------------------------------------------------------
def test_sap_client_btp_oauth_authentication(sap_client):
    """Test SAP BTP OAuth 2.0 client token grant."""
    auth = sap_client.authenticate_btp_oauth()
    assert auth["auth_status"] == "AUTHENTICATED"
    assert "Bearer" in auth["token_type"]
    assert "destination_name" in auth


def test_sap_client_create_maintenance_notification(sap_client):
    """Test dispatch of SAP PM Maintenance Notification."""
    payload = {
        "NotificationType": "M1",
        "Equipment": "MOTOR-001",
        "Priority": "2",
        "NotificationText": "AI Diagnostic Alert: Bearing Wear on MOTOR-001",
        "ReportedByUser": "AGENTIC_SWARM_AI",
    }
    resp = sap_client.create_maintenance_notification(payload)
    assert resp["status_code"] == 201
    assert resp["success"] is True
    assert "notification_id" in resp
    assert resp["notification_id"] in sap_client.ledger_notifications


def test_sap_client_create_maintenance_order(sap_client):
    """Test dispatch of SAP PM Maintenance Work Order with operations."""
    payload = {
        "MaintenanceOrderType": "PM01",
        "Equipment": "MOTOR-001",
        "Priority": "1",
        "MainWorkCenter": "MECH-01",
        "to_Operations": [
            {"Operation": "0010", "OperationText": "Lockout/Tagout (LOTO)"},
            {"Operation": "0020", "OperationText": "Replace SKF 6205-2RS Bearing"},
        ],
        "to_Components": [
            {"Material": "MAT-BRG-6205-2RS", "RequirementQuantity": 1.0},
        ],
    }
    resp = sap_client.create_maintenance_order(payload)
    assert resp["status_code"] == 201
    assert resp["success"] is True
    assert resp["operations_count"] == 2
    assert resp["components_count"] == 1

    order_status = sap_client.get_order_status(resp["order_id"])
    assert order_status["exists"] is True
    assert order_status["status"] == "RELEASED"


def test_sap_client_create_purchase_requisition(sap_client):
    """Test dispatch of SAP MM Purchase Requisition."""
    payload = {
        "PurchaseRequisitionType": "NB",
        "EstimatedTotalValueUSD": 42.50,
        "to_PurchaseRequisitionItem": [
            {
                "PurchaseRequisitionItem": "00010",
                "Material": "MAT-BRG-6205-2RS",
                "RequestedQuantity": 1.0,
                "Plant": "PLANT-01",
                "StorageLocation": "SL01",
            }
        ],
    }
    resp = sap_client.create_purchase_requisition(payload)
    assert resp["status_code"] == 201
    assert resp["success"] is True
    assert resp["total_items"] == 1

    pr_status = sap_client.get_pr_status(resp["purchase_requisition_id"])
    assert pr_status["exists"] is True
    assert pr_status["status"] == "SUBMITTED"


# -----------------------------------------------------------------------------
# 4. SAP Fiori Dashboard Adapter Tests
# -----------------------------------------------------------------------------
def test_fiori_dashboard_adapter_card_generation(digital_twin):
    """Test generation of analytical, KPI, prognostics, and ERP Fiori cards."""
    # Populate twin with sample warning report
    report = DiagnosticReport(
        asset_id="MOTOR-001",
        timestamp="2026-10-25T14:30:00Z",
        is_anomaly=True,
        anomaly_score=0.75,
        risk_level="HIGH",
        predicted_class=FaultClass.BEARING_WEAR,
        predicted_failure_mode="Bearing Wear (BPFO)",
        confidence=0.88,
        rul_hours=32.0,
        reconstruction_error=0.035,
        isolation_forest_score=0.72,
        evidence=["Envelope peak at 107.4 Hz"],
        recommended_action="Replace SKF 6205-2RS",
        features={"rad_rms": 0.78, "temp_mean": 60.0, "current_mean": 1.25},
    )
    digital_twin.update_telemetry(report)

    # 1. Gauge Card
    gauge = FioriDashboardAdapter.get_health_gauge_card(digital_twin)
    assert gauge["card_type"] == "AnalyticalCard"
    assert gauge["value"] == digital_twin.health_score
    assert "color_code" in gauge

    # 2. Telemetry KPI Card
    telemetry = FioriDashboardAdapter.get_telemetry_card(digital_twin)
    assert telemetry["card_type"] == "NumericCard"
    assert len(telemetry["metrics"]) == 4

    # 3. Prognostics Card
    prognostics = FioriDashboardAdapter.get_prognostics_card(digital_twin)
    assert prognostics["card_type"] == "ListCard"
    assert prognostics["predicted_failure_mode"] == "Bearing Wear (BPFO)"

    # 4. Full Snapshot
    snapshot = FioriDashboardAdapter.get_full_dashboard_snapshot(
        twin=digital_twin,
        recent_order={"order_id": "40003001", "type": "PM01"},
    )
    assert "fiori_cards" in snapshot
    assert len(snapshot["fiori_cards"]) == 4
    assert len(snapshot["state_transitions"]) >= 1


# -----------------------------------------------------------------------------
# 5. Full Closed-Loop Integration Test
# -----------------------------------------------------------------------------
def test_full_closed_loop_physics_to_sap_and_fiori_maintenance():
    """
    End-to-End Enterprise Journey:
      1. Physical Asset Simulator produces degraded bearing telemetry
      2. Multimodal Prognostics Engine predicts failure mode & RUL
      3. Digital Twin updates health score and transitions state to WARNING
      4. LangGraph Multi-Agent Swarm formulates diagnosis, checks MM inventory, and generates SAP PM/MM payloads
      5. SAP Client dispatches Work Order to S/4HANA OData v4 API
      6. Digital Twin records active Work Order and transitions state to MAINTENANCE
      7. Fiori Dashboard reflects live maintenance order status
    """
    sim = PhysicalSignalSimulator()
    engine = PrognosticsEngine()
    twin = DigitalTwin()
    swarm = DecisionSwarm()
    sap = SAPClient()

    # Step 1: Simulate degraded vibration window
    batch = sim.generate_window(FaultClass.BEARING_WEAR, duration_sec=1.0, seed=101)

    # Step 2: Prognostics Inference
    report: DiagnosticReport = engine.predict_batch(batch)
    assert report.is_anomaly is True
    assert report.predicted_class == FaultClass.BEARING_WEAR

    # Step 3: Digital Twin Health Score & State Update
    state = twin.update_telemetry(report)
    assert state in (AssetState.WARNING, AssetState.CRITICAL)
    assert twin.health_score < 70.0

    # Step 4: Multi-Agent Decision Swarm
    swarm_result = swarm.run(report)
    assert swarm_result["swarm_status"] == "COMPLETED"

    # Step 5: SAP OData Work Order & Notification Dispatch
    notif_resp = sap.create_maintenance_notification(swarm_result["sap_pm_notification"])
    order_resp = sap.create_maintenance_order(swarm_result["sap_pm_order"])
    assert notif_resp["success"] is True
    assert order_resp["success"] is True

    # Step 6: Transition Digital Twin to MAINTENANCE State
    twin.set_maintenance_active(order_resp["order_id"])
    assert twin.state == AssetState.MAINTENANCE
    assert order_resp["order_id"] in twin.active_work_orders

    # Step 7: Fiori Dashboard Status Snapshot
    snapshot = FioriDashboardAdapter.get_full_dashboard_snapshot(
        twin=twin,
        recent_order=order_resp,
    )
    assert snapshot["asset_overview"]["state"] == "MAINTENANCE"
    assert snapshot["asset_overview"]["active_work_order"] == order_resp["order_id"]
    assert "ACTIVE MAINTENANCE" in snapshot["fiori_cards"][3]["status_banner"]
