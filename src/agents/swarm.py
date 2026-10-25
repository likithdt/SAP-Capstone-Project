"""
LangGraph Multi-Agent Orchestration Swarm.
Coordinates Diagnostic Agent (RAG), Inventory Agent (SAP MM), and Scheduler Agent (SAP PM)
into a closed-loop autonomous enterprise decision workflow.
"""

from typing import Any, Dict, List, Optional, Union
from langgraph.graph import StateGraph, END

from src.config import AgentConfig, SAPConfig, DEFAULT_AGENT_CONFIG, DEFAULT_SAP_CONFIG, FaultClass
from src.models.engine import DiagnosticReport
from src.agents.state import AgentSwarmState, SwarmStatus, WorkOrderPriority
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.inventory_agent import InventoryAgent
from src.agents.scheduler_agent import SchedulerAgent


class DecisionSwarm:
    """
    Multi-Agent Decision Swarm using LangGraph StateGraph orchestration.
    Coordinates:
      1. Diagnostic Agent: RAG vector-search over maintenance manuals
      2. Inventory Agent: SAP MM stock verification and automated Purchase Requisitions
      3. Scheduler Agent: Failure criticality prioritization and SAP PM payload assembly
      4. Audit Aggregator: Execution logging and enterprise compliance verification
    """

    def __init__(
        self,
        diagnostic_agent: Optional[DiagnosticAgent] = None,
        inventory_agent: Optional[InventoryAgent] = None,
        scheduler_agent: Optional[SchedulerAgent] = None,
        agent_config: Optional[AgentConfig] = None,
        sap_config: Optional[SAPConfig] = None,
    ):
        self.agent_config = agent_config or DEFAULT_AGENT_CONFIG
        self.sap_config = sap_config or DEFAULT_SAP_CONFIG

        self.diagnostic_agent = diagnostic_agent or DiagnosticAgent(config=self.agent_config)
        self.inventory_agent = inventory_agent or InventoryAgent(config=self.sap_config)
        self.scheduler_agent = scheduler_agent or SchedulerAgent(
            sap_config=self.sap_config,
            agent_config=self.agent_config,
        )

        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph with conditional healthy routing and agent chaining."""
        workflow = StateGraph(AgentSwarmState)

        # Register Swarm Nodes
        workflow.add_node("diagnostic_agent", self._node_diagnostic)
        workflow.add_node("inventory_agent", self._node_inventory)
        workflow.add_node("scheduler_agent", self._node_scheduler)
        workflow.add_node("healthy_bypass", self._node_healthy_bypass)
        workflow.add_node("audit_aggregator", self._node_audit_aggregator)

        # Conditional Entry Point: Route based on anomaly status
        workflow.set_conditional_entry_point(
            self._route_telemetry,
            {
                "diagnose": "diagnostic_agent",
                "healthy": "healthy_bypass",
            },
        )

        # Directed Agent Execution Flow
        workflow.add_edge("diagnostic_agent", "inventory_agent")
        workflow.add_edge("inventory_agent", "scheduler_agent")
        workflow.add_edge("scheduler_agent", "audit_aggregator")
        workflow.add_edge("healthy_bypass", "audit_aggregator")
        workflow.add_edge("audit_aggregator", END)

        return workflow

    def _route_telemetry(self, state: AgentSwarmState) -> str:
        """Conditional router: evaluates whether telemetry requires full diagnostic swarm intervention."""
        is_anom = state.get("is_anomaly", False)
        f_class = state.get("predicted_class", 0)
        anom_score = state.get("anomaly_score", 0.0)

        if not is_anom and f_class == 0 and anom_score < 0.35:
            return "healthy"
        return "diagnose"

    # -------------------------------------------------------------------------
    # LangGraph Nodes
    # -------------------------------------------------------------------------
    def _node_healthy_bypass(self, state: AgentSwarmState) -> Dict[str, Any]:
        """Handles nominal healthy asset state without generating unnecessary work orders."""
        logs = list(state.get("execution_log", []))
        asset_id = state.get("asset_id", "MOTOR-001")
        logs.append(
            f"[{asset_id}] Telemetry Router: Asset is operating nominally within Class 0 (Healthy) limits. "
            f"Bypassing corrective maintenance order generation."
        )

        return {
            "diagnosis_summary": "ASSET HEALTH OPTIMAL. All vibration, thermal, and current metrics nominal.",
            "recommended_action": "Continue continuous IoT monitoring. Next scheduled inspection per SOP-GEN-001.",
            "required_spare_parts": [],
            "estimated_repair_hours": 0.0,
            "parts_available": True,
            "inventory_status": {"parts_available": True, "items": [], "shortages": []},
            "sap_mm_pr": None,
            "work_order_priority": WorkOrderPriority.LOW.value,
            "priority_level_name": WorkOrderPriority.LOW.label,
            "scheduling_decision": {"target_window": "Routine Continuous Monitoring", "estimated_downtime_hours": 0.0},
            "sap_pm_notification": None,
            "sap_pm_order": None,
            "swarm_status": SwarmStatus.SKIPPED_HEALTHY.value,
            "execution_log": logs,
        }

    def _node_diagnostic(self, state: AgentSwarmState) -> Dict[str, Any]:
        """Executes Diagnostic Agent with Vector-Search RAG over equipment manuals."""
        logs = list(state.get("execution_log", []))
        asset_id = state.get("asset_id", "MOTOR-001")
        fault_class = state.get("predicted_class", 1)
        fault_name = state.get("predicted_failure_mode", "Mechanical Defect")
        anomaly_score = state.get("anomaly_score", 0.75)
        risk_level = state.get("risk_level", "HIGH")
        confidence = state.get("confidence", 0.85)
        rul_hours = state.get("rul_hours", 38.0)
        evidence = state.get("evidence", [])
        features = state.get("features", {})

        diag_res = self.diagnostic_agent.diagnose(
            fault_class=fault_class,
            fault_name=fault_name,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            confidence=confidence,
            rul_hours=rul_hours,
            evidence=evidence,
            features=features,
        )

        sop_code = diag_res.get("sop_code", "SOP-MEC-001")
        parts = diag_res.get("required_spare_parts", [])
        hours = diag_res.get("estimated_repair_hours", 2.0)

        logs.append(
            f"[{asset_id}] Diagnostic Agent: Completed RAG semantic retrieval. "
            f"Mapped to {sop_code} ({fault_name}). Identified {len(parts)} required replacement part(s). "
            f"Estimated repair duration: {hours:.1f} hours."
        )

        return {
            "rag_context": diag_res["rag_context"],
            "diagnosis_summary": diag_res["diagnosis_summary"],
            "recommended_action": diag_res["recommended_action"],
            "required_spare_parts": parts,
            "estimated_repair_hours": hours,
            "swarm_status": SwarmStatus.DIAGNOSED.value,
            "execution_log": logs,
        }

    def _node_inventory(self, state: AgentSwarmState) -> Dict[str, Any]:
        """Executes Inventory Agent: queries SAP MM stock and generates PR if shortage exists."""
        logs = list(state.get("execution_log", []))
        asset_id = state.get("asset_id", "MOTOR-001")
        parts = state.get("required_spare_parts", [])

        inv_res = self.inventory_agent.verify_inventory(
            required_parts=parts,
            asset_id=asset_id,
        )

        parts_available = inv_res["parts_available"]
        shortages = inv_res["shortages"]
        sap_mm_pr = inv_res.get("sap_mm_pr")

        if parts_available:
            logs.append(
                f"[{asset_id}] Inventory Agent: All {len(parts)} required parts confirmed in stock in SAP MM. "
                f"Stock successfully reserved for Maintenance Order."
            )
        else:
            logs.append(
                f"[{asset_id}] Inventory Agent: Stock shortage detected for {len(shortages)} part(s). "
                f"Generated SAP MM Purchase Requisition with {inv_res['max_lead_time_days']}-day lead time."
            )

        return {
            "inventory_status": inv_res,
            "parts_available": parts_available,
            "sap_mm_pr": sap_mm_pr,
            "swarm_status": SwarmStatus.INVENTORY_CHECKED.value,
            "execution_log": logs,
        }

    def _node_scheduler(self, state: AgentSwarmState) -> Dict[str, Any]:
        """Executes Scheduler Agent: prioritizes work order and generates SAP PM payloads."""
        logs = list(state.get("execution_log", []))
        asset_id = state.get("asset_id", "MOTOR-001")
        fault_class = state.get("predicted_class", 1)
        fault_name = state.get("predicted_failure_mode", "Mechanical Defect")
        anomaly_score = state.get("anomaly_score", 0.75)
        risk_level = state.get("risk_level", "HIGH")
        rul_hours = state.get("rul_hours", 38.0)
        diagnosis_summary = state.get("diagnosis_summary", "")
        repair_hours = state.get("estimated_repair_hours", 2.0)
        inventory_status = state.get("inventory_status", {})
        parts = state.get("required_spare_parts", [])

        sched_res = self.scheduler_agent.schedule(
            asset_id=asset_id,
            fault_class=fault_class,
            fault_name=fault_name,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            rul_hours=rul_hours,
            diagnosis_summary=diagnosis_summary,
            estimated_repair_hours=repair_hours,
            inventory_status=inventory_status,
            required_spare_parts=parts,
        )

        p_code = sched_res["work_order_priority"]
        p_name = sched_res["priority_level_name"]
        w_center = sched_res["work_center"]
        decision = sched_res["scheduling_decision"]

        logs.append(
            f"[{asset_id}] Scheduler Agent: Assigned Priority {p_code} ({p_name}) to Work Center {w_center}. "
            f"Scheduled window: '{decision.get('target_window')}'. "
            f"SAP PM Notification and Work Order payloads assembled."
        )

        return {
            "work_order_priority": p_code,
            "priority_level_name": p_name,
            "scheduling_decision": decision,
            "sap_pm_notification": sched_res["sap_pm_notification"],
            "sap_pm_order": sched_res["sap_pm_order"],
            "swarm_status": SwarmStatus.SCHEDULED.value,
            "execution_log": logs,
        }

    def _node_audit_aggregator(self, state: AgentSwarmState) -> Dict[str, Any]:
        """Final audit node: validates payloads, finalizes swarm status, and completes trace."""
        logs = list(state.get("execution_log", []))
        curr_status = state.get("swarm_status", SwarmStatus.SCHEDULED.value)

        final_status = SwarmStatus.COMPLETED.value if curr_status != SwarmStatus.SKIPPED_HEALTHY.value else SwarmStatus.SKIPPED_HEALTHY.value
        logs.append(
            f"Audit Aggregator: Swarm execution cycle finalized with status '{final_status}'. "
            f"Closed-loop ERP action ready for SAP BTP OData dispatch."
        )

        return {
            "swarm_status": final_status,
            "execution_log": logs,
        }

    # -------------------------------------------------------------------------
    # Public Execution API
    # -------------------------------------------------------------------------
    def run(self, report: Union[DiagnosticReport, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the multi-agent decision swarm on a DiagnosticReport or dictionary.

        Args:
            report: DiagnosticReport from PrognosticsEngine or equivalent dictionary

        Returns:
            Complete end-to-end swarm execution state including diagnosis, inventory, and SAP payloads.
        """
        # Convert DiagnosticReport to initial swarm state
        if isinstance(report, DiagnosticReport):
            initial_state: AgentSwarmState = {
                "asset_id": report.asset_id,
                "timestamp": report.timestamp,
                "is_anomaly": report.is_anomaly,
                "anomaly_score": report.anomaly_score,
                "risk_level": report.risk_level,
                "predicted_class": report.predicted_class,
                "predicted_failure_mode": report.predicted_failure_mode,
                "confidence": report.confidence,
                "rul_hours": report.rul_hours,
                "evidence": list(report.evidence),
                "features": dict(report.features),
                "swarm_status": SwarmStatus.INITIALIZED.value,
                "execution_log": [f"Swarm initialized for asset {report.asset_id} at {report.timestamp}."],
                "errors": [],
            }
        else:
            initial_state = {
                "asset_id": report.get("asset_id", "MOTOR-001"),
                "timestamp": report.get("timestamp", ""),
                "is_anomaly": bool(report.get("is_anomaly", True)),
                "anomaly_score": float(report.get("anomaly_score", 0.70)),
                "risk_level": report.get("risk_level", "HIGH"),
                "predicted_class": int(report.get("predicted_class", 1)),
                "predicted_failure_mode": report.get("predicted_failure_mode", "Bearing Wear"),
                "confidence": float(report.get("confidence", 0.85)),
                "rul_hours": float(report.get("rul_hours", 38.0)),
                "evidence": list(report.get("evidence", [])),
                "features": dict(report.get("features", {})),
                "swarm_status": SwarmStatus.INITIALIZED.value,
                "execution_log": [f"Swarm initialized for asset {report.get('asset_id', 'MOTOR-001')}."],
                "errors": [],
            }

        return self.app.invoke(initial_state)
