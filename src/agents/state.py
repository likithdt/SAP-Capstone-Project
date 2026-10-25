"""
LangGraph Multi-Agent Swarm State Schema and Supporting Data Models.
Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine with Closed-Loop ERP Action.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class SwarmStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    DIAGNOSED = "DIAGNOSED"
    INVENTORY_CHECKED = "INVENTORY_CHECKED"
    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    SKIPPED_HEALTHY = "SKIPPED_HEALTHY"
    FAILED = "FAILED"


class WorkOrderPriority(str, Enum):
    VERY_HIGH_EMERGENCY = "1"  # RUL < 12h or critical fault
    HIGH = "2"                 # RUL 12-48h
    MEDIUM = "3"               # RUL 48-100h
    LOW = "4"                  # RUL > 100h / routine preventive maintenance

    @property
    def label(self) -> str:
        labels = {
            WorkOrderPriority.VERY_HIGH_EMERGENCY: "Emergency / Very High (< 12h)",
            WorkOrderPriority.HIGH: "High Priority (< 24h - 48h)",
            WorkOrderPriority.MEDIUM: "Medium Priority (2 - 4 Days)",
            WorkOrderPriority.LOW: "Low / Planned Overhaul (> 4 Days)",
        }
        return labels[self]


class SparePartRequirement(BaseModel):
    """Specification of a spare part identified by the Diagnostic Agent."""
    material_id: str = Field(description="SAP Material Number e.g. MAT-BRG-6205-2RS")
    material_name: str = Field(description="Standard commercial equipment part description")
    quantity_required: int = Field(default=1, description="Quantity required for repair")
    unit_of_measure: str = Field(default="EA", description="SAP Base Unit of Measure")
    is_critical: bool = Field(default=True, description="Whether missing this part blocks repair")


class InventoryItemStatus(BaseModel):
    """Status of an individual spare part in SAP MM."""
    material_id: str
    material_name: str
    quantity_required: int
    quantity_on_hand: int
    quantity_available: int
    is_in_stock: bool
    lead_time_days: int
    storage_location: str = "SL01"
    unit_cost_usd: float = 0.0


class AgentSwarmState(TypedDict, total=False):
    """
    Unified LangGraph State Schema passed across all swarm agents:
    Diagnostic Agent -> Inventory Agent -> Scheduler Agent -> Audit Aggregator.
    """
    # 1. Telemetry and Prognostics Input
    asset_id: str
    timestamp: str
    is_anomaly: bool
    anomaly_score: float
    risk_level: str
    predicted_class: int
    predicted_failure_mode: str
    confidence: float
    rul_hours: float
    evidence: List[str]
    features: Dict[str, Any]

    # 2. Diagnostic Agent with RAG Outputs
    rag_context: List[Dict[str, Any]]
    diagnosis_summary: str
    recommended_action: str
    required_spare_parts: List[Dict[str, Any]]
    estimated_repair_hours: float

    # 3. Inventory Agent Outputs (SAP MM)
    inventory_status: Dict[str, Any]
    parts_available: bool
    sap_mm_pr: Optional[Dict[str, Any]]

    # 4. Scheduler Agent Outputs (SAP PM)
    work_order_priority: str
    priority_level_name: str
    scheduling_decision: Dict[str, Any]
    sap_pm_notification: Optional[Dict[str, Any]]
    sap_pm_order: Optional[Dict[str, Any]]

    # 5. Swarm Orchestration & Audit Trace
    swarm_status: str
    execution_log: List[str]
    errors: List[str]
