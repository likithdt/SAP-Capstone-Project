"""
Comprehensive Test Suite for Step 5: Agentic AI Decision Swarm.
Validates:
  - ChromaDB Semantic Vector-Search RAG over Maintenance Manuals (Diagnostic Agent)
  - SAP MM Material Availability Verification & Automated PR Payloads (Inventory Agent)
  - Criticality Prioritization, Shift Windows & SAP PM Work Orders (Scheduler Agent)
  - Full LangGraph Multi-Agent Orchestration & Physics-to-ERP Closed Loop
"""

import pytest
from src.config import AssetConfig, FaultClass, SAPConfig, AgentConfig
from src.digital_twin.simulator import PhysicalSignalSimulator
from src.models.engine import PrognosticsEngine, DiagnosticReport
from src.agents.state import SwarmStatus, WorkOrderPriority
from src.agents.rag.knowledge_base import EQUIPMENT_MAINTENANCE_MANUALS
from src.agents.rag.vector_store import MaintenanceVectorStore
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.inventory_agent import InventoryAgent
from src.agents.scheduler_agent import SchedulerAgent
from src.agents.swarm import DecisionSwarm


@pytest.fixture
def vector_store():
    """In-memory ChromaDB vector store fixture."""
    return MaintenanceVectorStore(in_memory=True)


@pytest.fixture
def diagnostic_agent(vector_store):
    return DiagnosticAgent(vector_store=vector_store)


@pytest.fixture
def inventory_agent():
    return InventoryAgent()


@pytest.fixture
def scheduler_agent():
    return SchedulerAgent()


@pytest.fixture
def decision_swarm(diagnostic_agent, inventory_agent, scheduler_agent):
    return DecisionSwarm(
        diagnostic_agent=diagnostic_agent,
        inventory_agent=inventory_agent,
        scheduler_agent=scheduler_agent,
    )


# -----------------------------------------------------------------------------
# 1. RAG Vector Store Tests
# -----------------------------------------------------------------------------
def test_vector_store_seeding_and_retrieval(vector_store):
    """Test that all SOP documents are indexed and retrievable via semantic vector search."""
    assert vector_store.collection.count() == len(EQUIPMENT_MAINTENANCE_MANUALS)

    # Search for bearing wear using acoustic/vibration keywords
    results = vector_store.search("bearing raceway spalling BPFO 107.4 Hz kurtosis impact", n_results=2)
    assert len(results) > 0
    top = results[0]
    assert top["fault_class"] == FaultClass.BEARING_WEAR
    assert "SOP-MEC-001" in top["sop_code"]
    assert top["similarity_score"] > 0.40

    # Search for thermal overheating
    thermal_res = vector_store.search("motor overheating stator temperature 85 degrees fan failure", n_results=1)
    assert len(thermal_res) == 1
    assert thermal_res[0]["fault_class"] == FaultClass.THERMAL_OVERHEATING
    assert thermal_res[0]["sop_code"] == "SOP-ELE-001"


def test_vector_store_metadata_filtering(vector_store):
    """Test targeted metadata filtering by fault class."""
    results = vector_store.search("maintenance procedure", n_results=1, filter_metadata={"fault_class": 2})
    assert len(results) == 1
    assert results[0]["fault_class"] == FaultClass.ROTOR_IMBALANCE
    assert results[0]["sop_code"] == "SOP-MEC-002"


# -----------------------------------------------------------------------------
# 2. Diagnostic Agent Tests
# -----------------------------------------------------------------------------
def test_diagnostic_agent_bearing_wear(diagnostic_agent):
    """Test RAG diagnostic synthesis for rolling element bearing wear."""
    diag = diagnostic_agent.diagnose(
        fault_class=FaultClass.BEARING_WEAR,
        fault_name="Bearing Wear (BPFO)",
        anomaly_score=0.84,
        risk_level="HIGH",
        confidence=0.89,
        rul_hours=32.0,
        evidence=["Elevated kurtosis 5.2 (> 4.5)", "Prominent envelope peak at 107.4 Hz BPFO"],
    )

    assert "SOP-MEC-001" in diag["diagnosis_summary"]
    assert diag["estimated_repair_hours"] == 2.5
    assert len(diag["required_spare_parts"]) >= 1

    part_ids = [p["material_id"] for p in diag["required_spare_parts"]]
    assert "MAT-BRG-6205-2RS" in part_ids


def test_diagnostic_agent_thermal_overheating(diagnostic_agent):
    """Test RAG diagnostic synthesis for thermal overheating."""
    diag = diagnostic_agent.diagnose(
        fault_class=FaultClass.THERMAL_OVERHEATING,
        fault_name="Thermal Overheating",
        anomaly_score=0.91,
        risk_level="CRITICAL",
        confidence=0.94,
        rul_hours=14.0,
        evidence=["Stator temperature reached 86.5 °C (> 85 °C critical limit)"],
    )

    assert "SOP-ELE-001" in diag["diagnosis_summary"]
    assert diag["estimated_repair_hours"] == 2.0
    part_ids = [p["material_id"] for p in diag["required_spare_parts"]]
    assert "MAT-FAN-COOL-100" in part_ids


def test_diagnostic_agent_healthy(diagnostic_agent):
    """Test Diagnostic Agent handling of nominal healthy asset."""
    diag = diagnostic_agent.diagnose(
        fault_class=FaultClass.HEALTHY,
        fault_name="Healthy",
        anomaly_score=0.12,
        risk_level="LOW",
        confidence=0.96,
        rul_hours=120.0,
        evidence=["Vibration RMS nominal at 0.22 g"],
    )

    assert "OPTIMAL" in diag["diagnosis_summary"]
    assert len(diag["required_spare_parts"]) == 0
    assert "No maintenance action required" in diag["recommended_action"]


# -----------------------------------------------------------------------------
# 3. Inventory Agent Tests
# -----------------------------------------------------------------------------
def test_inventory_agent_in_stock(inventory_agent):
    """Test inventory verification when all required spare parts are in stock."""
    required = [
        {"material_id": "MAT-BRG-6205-2RS", "quantity_required": 1},
        {"material_id": "MAT-LUB-LGMT2", "quantity_required": 1},
    ]
    res = inventory_agent.verify_inventory(required, asset_id="MOTOR-001")

    assert res["parts_available"] is True
    assert len(res["shortages"]) == 0
    assert res["sap_mm_pr"] is None
    assert "SL01" in res["summary"]


def test_inventory_agent_out_of_stock_generates_pr(inventory_agent):
    """Test automated SAP MM Purchase Requisition generation when a component is out of stock."""
    # Force out of stock on bearing
    inventory_agent.set_stock("MAT-BRG-6205-2RS", quantity_on_hand=0)

    required = [{"material_id": "MAT-BRG-6205-2RS", "quantity_required": 1}]
    res = inventory_agent.verify_inventory(required, asset_id="MOTOR-001")

    assert res["parts_available"] is False
    assert len(res["shortages"]) == 1
    assert res["max_lead_time_days"] == 5

    pr = res["sap_mm_pr"]
    assert pr is not None
    assert pr["PurchaseRequisitionType"] == "NB"
    assert pr["Plant"] == "PLANT-01"
    assert pr["CostCenter"] == "CC-MAINT-01"
    assert len(pr["to_PurchaseRequisitionItem"]) == 1

    item = pr["to_PurchaseRequisitionItem"][0]
    assert item["Material"] == "MAT-BRG-6205-2RS"
    assert item["RequestedQuantity"] == 1.0
    assert item["PurchaseRequisitionPrice"] == 42.50
    assert "DeliveryDate" in item


def test_inventory_agent_empty_requirements(inventory_agent):
    """Test inventory check with no required parts (e.g. healthy asset)."""
    res = inventory_agent.verify_inventory([])
    assert res["parts_available"] is True
    assert res["sap_mm_pr"] is None


# -----------------------------------------------------------------------------
# 4. Scheduler Agent Tests
# -----------------------------------------------------------------------------
def test_scheduler_agent_emergency_priority(scheduler_agent):
    """Test Priority 1 (Emergency) assignment for short RUL or critical risk."""
    inv_status = {"parts_available": True, "max_lead_time_days": 0}
    res = scheduler_agent.schedule(
        asset_id="MOTOR-001",
        fault_class=FaultClass.MOTOR_OVERLOAD,
        fault_name="Motor Overload",
        anomaly_score=0.92,
        risk_level="CRITICAL",
        rul_hours=6.5,
        diagnosis_summary="Conveyor mechanical jam inducing 3.4 A current draw",
        estimated_repair_hours=1.0,
        inventory_status=inv_status,
        required_spare_parts=[{"material_id": "MAT-BELT-V-A42", "quantity_required": 1}],
    )

    assert res["work_order_priority"] == "1"
    assert "Emergency" in res["priority_level_name"]
    assert res["work_center"] == "MECH-01"

    # Verify SAP PM Notification Payload
    notif = res["sap_pm_notification"]
    assert notif["NotificationType"] == "M1"
    assert notif["Priority"] == "1"
    assert notif["BreakdownIndicator"] is True
    assert notif["Equipment"] == "MOTOR-001"

    # Verify SAP PM Work Order Payload
    order = res["sap_pm_order"]
    assert order["MaintenanceOrderType"] == "PM01"
    assert order["Priority"] == "1"
    assert len(order["to_Operations"]) == 5
    assert len(order["to_Components"]) == 1


def test_scheduler_agent_high_and_medium_priority(scheduler_agent):
    """Test Priority 2 (High) and Priority 3 (Medium) scheduling."""
    inv_status = {"parts_available": True, "max_lead_time_days": 0}

    # Priority 2: RUL = 36h
    res_p2 = scheduler_agent.schedule(
        asset_id="MOTOR-001",
        fault_class=FaultClass.BEARING_WEAR,
        fault_name="Bearing Wear",
        anomaly_score=0.72,
        risk_level="HIGH",
        rul_hours=36.0,
        diagnosis_summary="BPFO spalling",
        estimated_repair_hours=2.5,
        inventory_status=inv_status,
    )
    assert res_p2["work_order_priority"] == "2"

    # Priority 3: RUL = 72h
    res_p3 = scheduler_agent.schedule(
        asset_id="MOTOR-001",
        fault_class=FaultClass.ROTOR_IMBALANCE,
        fault_name="Rotor Imbalance",
        anomaly_score=0.55,
        risk_level="MEDIUM",
        rul_hours=72.0,
        diagnosis_summary="1X dynamic imbalance",
        estimated_repair_hours=1.5,
        inventory_status=inv_status,
    )
    assert res_p3["work_order_priority"] == "3"


def test_scheduler_agent_supply_bottleneck(scheduler_agent):
    """Test supply chain bottleneck alert when lead time exceeds remaining useful life."""
    # Lead time = 7 days = 168 hours > RUL (24 hours)
    inv_status = {"parts_available": False, "max_lead_time_days": 7}
    res = scheduler_agent.schedule(
        asset_id="MOTOR-001",
        fault_class=FaultClass.BEARING_WEAR,
        fault_name="Bearing Wear",
        anomaly_score=0.85,
        risk_level="HIGH",
        rul_hours=24.0,
        diagnosis_summary="Bearing wear",
        estimated_repair_hours=2.5,
        inventory_status=inv_status,
    )

    timeline = res["scheduling_decision"]
    assert timeline["supply_constraint_alert"] is True
    assert "CRITICAL SUPPLY BOTTLENECK" in timeline["operational_mitigation"]


# -----------------------------------------------------------------------------
# 5. LangGraph Decision Swarm End-to-End Tests
# -----------------------------------------------------------------------------
def test_decision_swarm_bearing_wear_full_cycle(decision_swarm):
    """Test complete LangGraph multi-agent swarm workflow on Bearing Wear."""
    report = {
        "asset_id": "MOTOR-001",
        "timestamp": "2026-10-25T14:00:00Z",
        "is_anomaly": True,
        "anomaly_score": 0.81,
        "risk_level": "HIGH",
        "predicted_class": FaultClass.BEARING_WEAR,
        "predicted_failure_mode": "Bearing Wear (BPFO)",
        "confidence": 0.88,
        "rul_hours": 34.0,
        "evidence": ["Kurtosis elevated to 5.1", "107.4 Hz BPFO harmonic detected"],
        "features": {"rad_rms": 0.88, "rad_kurtosis": 5.1},
    }

    result = decision_swarm.run(report)

    assert result["swarm_status"] == SwarmStatus.COMPLETED.value
    assert len(result["rag_context"]) > 0
    assert "SOP-MEC-001" in result["diagnosis_summary"]
    assert result["parts_available"] is True
    assert result["work_order_priority"] == "2"
    assert result["sap_pm_notification"] is not None
    assert result["sap_pm_order"] is not None

    # Check execution log audit trail
    logs = result["execution_log"]
    assert any("Diagnostic Agent" in l for l in logs)
    assert any("Inventory Agent" in l for l in logs)
    assert any("Scheduler Agent" in l for l in logs)
    assert any("Audit Aggregator" in l for l in logs)


def test_decision_swarm_healthy_bypass_cycle(decision_swarm):
    """Test LangGraph conditional router bypassing work order creation for healthy assets."""
    healthy_report = {
        "asset_id": "MOTOR-001",
        "timestamp": "2026-10-25T14:00:00Z",
        "is_anomaly": False,
        "anomaly_score": 0.14,
        "risk_level": "LOW",
        "predicted_class": FaultClass.HEALTHY,
        "predicted_failure_mode": "Healthy",
        "confidence": 0.96,
        "rul_hours": 120.0,
        "evidence": ["Nominal vibration 0.21 g RMS"],
        "features": {"rad_rms": 0.21},
    }

    result = decision_swarm.run(healthy_report)

    assert result["swarm_status"] == SwarmStatus.SKIPPED_HEALTHY.value
    assert result["sap_pm_order"] is None
    assert result["sap_pm_notification"] is None
    assert result["sap_mm_pr"] is None
    assert "OPTIMAL" in result["diagnosis_summary"]


def test_decision_swarm_out_of_stock_generates_pr(decision_swarm, inventory_agent):
    """Test LangGraph swarm generating automated SAP MM PR when spare parts are depleted."""
    inventory_agent.set_stock("MAT-FAN-COOL-100", quantity_on_hand=0)

    overheating_report = {
        "asset_id": "MOTOR-001",
        "timestamp": "2026-10-25T14:00:00Z",
        "is_anomaly": True,
        "anomaly_score": 0.90,
        "risk_level": "CRITICAL",
        "predicted_class": FaultClass.THERMAL_OVERHEATING,
        "predicted_failure_mode": "Thermal Overheating",
        "confidence": 0.93,
        "rul_hours": 11.0,
        "evidence": ["Stator temperature reached 87.0 °C"],
        "features": {"temp_max": 87.0},
    }

    result = decision_swarm.run(overheating_report)

    assert result["swarm_status"] == SwarmStatus.COMPLETED.value
    assert result["parts_available"] is False
    assert result["sap_mm_pr"] is not None
    assert result["sap_mm_pr"]["PurchaseRequisitionType"] == "NB"
    assert result["work_order_priority"] == "1"


def test_decision_swarm_end_to_end_physics_to_erp_integration():
    """
    Complete closed-loop verification:
    Physical Signal Simulator -> Feature Pipeline -> Prognostics Inference Engine ->
    LangGraph Multi-Agent Decision Swarm -> SAP PM/MM ERP Payload Generation.
    """
    sim = PhysicalSignalSimulator()
    engine = PrognosticsEngine()
    swarm = DecisionSwarm()

    # Generate synthetic degraded bearing telemetry window
    batch = sim.generate_window(FaultClass.BEARING_WEAR, duration_sec=1.0, seed=42)

    # 1. Run Prognostics Inference Engine
    report: DiagnosticReport = engine.predict_batch(batch)
    assert report.is_anomaly is True
    assert report.predicted_class == FaultClass.BEARING_WEAR
    assert report.rul_hours < 50.0

    # 2. Run LangGraph Multi-Agent Swarm on Prognostic Output
    swarm_result = swarm.run(report)

    # 3. Verify closed-loop ERP actions
    assert swarm_result["swarm_status"] == SwarmStatus.COMPLETED.value
    assert "SOP-MEC-001" in swarm_result["diagnosis_summary"]
    assert swarm_result["sap_pm_notification"]["NotificationType"] == "M1"
    assert swarm_result["sap_pm_order"]["MaintenanceOrderType"] == "PM01"
    assert len(swarm_result["sap_pm_order"]["to_Operations"]) == 5
    assert len(swarm_result["execution_log"]) >= 5
