"""
Scheduler Agent for Maintenance Prioritization & SAP PM Work Order Generation.
Evaluates failure criticality, Remaining Useful Life (RUL), shift windows, and parts availability
to prioritize maintenance activities and assemble compliant SAP PM Notification and Work Order payloads.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from src.config import SAPConfig, AgentConfig, DEFAULT_SAP_CONFIG, DEFAULT_AGENT_CONFIG, FaultClass
from src.agents.state import WorkOrderPriority


class SchedulerAgent:
    """
    Autonomous Maintenance Scheduler Agent.
    Prioritizes interventions based on physical degradation pace and production schedules,
    and produces fully-structured SAP S/4HANA PM payloads (`API_MAINTENANCEORDER`).
    """

    def __init__(
        self,
        sap_config: Optional[SAPConfig] = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        self.sap_config = sap_config or DEFAULT_SAP_CONFIG
        self.agent_config = agent_config or DEFAULT_AGENT_CONFIG

    def schedule(
        self,
        asset_id: str,
        fault_class: int,
        fault_name: str,
        anomaly_score: float,
        risk_level: str,
        rul_hours: float,
        diagnosis_summary: str,
        estimated_repair_hours: float,
        inventory_status: Dict[str, Any],
        required_spare_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate criticality, resolve production shift windows, and build SAP PM payloads.
        """
        parts_list = required_spare_parts or []

        # 1. Determine SAP Work Order Priority Code (1 = Emergency to 4 = Low)
        priority_enum = self._calculate_priority(
            fault_class=fault_class,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            rul_hours=rul_hours,
        )
        priority_code = priority_enum.value
        priority_label = priority_enum.label

        # 2. Select appropriate Work Center (MECH-01 or ELEC-01)
        work_center = self._determine_work_center(fault_class)

        # 3. Resolve Scheduling Windows against RUL & Inventory Lead Times
        parts_available = inventory_status.get("parts_available", True)
        lead_time_days = int(inventory_status.get("max_lead_time_days", 0))
        lead_time_hours = lead_time_days * 24.0

        scheduling_decision = self._compute_schedule_timeline(
            priority=priority_enum,
            rul_hours=rul_hours,
            repair_hours=estimated_repair_hours,
            parts_available=parts_available,
            lead_time_hours=lead_time_hours,
        )

        # 4. Construct SAP PM Maintenance Notification Payload
        sap_pm_notification = self._build_notification_payload(
            asset_id=asset_id,
            fault_class=fault_class,
            fault_name=fault_name,
            anomaly_score=anomaly_score,
            rul_hours=rul_hours,
            priority_code=priority_code,
            priority_label=priority_label,
            work_center=work_center,
            diagnosis_summary=diagnosis_summary,
            repair_hours=estimated_repair_hours,
        )

        # 5. Construct SAP PM Maintenance Work Order Payload
        sap_pm_order = self._build_work_order_payload(
            asset_id=asset_id,
            fault_class=fault_class,
            fault_name=fault_name,
            priority_code=priority_code,
            priority_label=priority_label,
            work_center=work_center,
            scheduling=scheduling_decision,
            repair_hours=estimated_repair_hours,
            required_parts=parts_list,
            inventory_status=inventory_status,
        )

        return {
            "work_order_priority": priority_code,
            "priority_level_name": priority_label,
            "work_center": work_center,
            "scheduling_decision": scheduling_decision,
            "sap_pm_notification": sap_pm_notification,
            "sap_pm_order": sap_pm_order,
        }

    def _calculate_priority(
        self,
        fault_class: int,
        anomaly_score: float,
        risk_level: str,
        rul_hours: float,
    ) -> WorkOrderPriority:
        """Map failure criticality, anomaly severity, and RUL to SAP priority 1-4."""
        if fault_class == FaultClass.HEALTHY:
            return WorkOrderPriority.LOW

        # Immediate Emergency: RUL < 12h or CRITICAL risk with high anomaly score
        if (
            rul_hours < self.agent_config.emergency_rul_threshold_hours
            or risk_level == "CRITICAL"
            or anomaly_score >= 0.85
            or fault_class == FaultClass.MOTOR_OVERLOAD  # Thermal runaway/stall risk
        ):
            return WorkOrderPriority.VERY_HIGH_EMERGENCY

        # High Priority: RUL 12 - 48h or HIGH risk
        if (
            rul_hours < self.agent_config.high_priority_rul_threshold_hours
            or risk_level == "HIGH"
            or anomaly_score >= 0.65
        ):
            return WorkOrderPriority.HIGH

        # Medium Priority: RUL 48 - 100h
        if (
            rul_hours < self.agent_config.medium_priority_rul_threshold_hours
            or risk_level == "MEDIUM"
            or anomaly_score >= 0.35
        ):
            return WorkOrderPriority.MEDIUM

        return WorkOrderPriority.LOW

    def _determine_work_center(self, fault_class: int) -> str:
        """Direct maintenance work to Mechanical or Electrical craft group."""
        electrical_classes = {
            FaultClass.THERMAL_OVERHEATING,
            FaultClass.SENSOR_ANOMALY,
        }
        if fault_class in electrical_classes:
            return self.sap_config.work_center_electrical
        return self.sap_config.work_center_mechanical

    def _compute_schedule_timeline(
        self,
        priority: WorkOrderPriority,
        rul_hours: float,
        repair_hours: float,
        parts_available: bool,
        lead_time_hours: float,
    ) -> Dict[str, Any]:
        """Compute planned start/finish times taking shift windows and parts lead time into account."""
        now = datetime.now(timezone.utc)

        # Shift calculations
        shift_duration = self.agent_config.shift_duration_hours

        supply_constraint_alert = False
        operational_mitigation = None

        if parts_available:
            if priority == WorkOrderPriority.VERY_HIGH_EMERGENCY:
                planned_start = now + timedelta(hours=1.0)
                target_window = "Immediate Emergency Window (< 2h)"
            elif priority == WorkOrderPriority.HIGH:
                planned_start = now + timedelta(hours=shift_duration)
                target_window = "Next Operational Shift (< 8h - 24h)"
            elif priority == WorkOrderPriority.MEDIUM:
                planned_start = now + timedelta(hours=48.0)
                target_window = "Planned Maintenance Window (48h - 72h)"
            else:
                planned_start = now + timedelta(days=7.0)
                target_window = "Scheduled Weekly Overhaul Window (> 7 Days)"
        else:
            # Parts are not on hand
            if lead_time_hours > rul_hours:
                supply_constraint_alert = True
                operational_mitigation = (
                    f"CRITICAL SUPPLY BOTTLENECK: Spare part lead time ({lead_time_hours:.0f}h) "
                    f"exceeds equipment RUL ({rul_hours:.1f}h). Immediate operational recommendation: "
                    f"Derate conveyor speed to 50% or initiate emergency expedited procurement."
                )
            # Schedule maintenance upon arrival of parts
            planned_start = now + timedelta(hours=max(lead_time_hours, 2.0))
            target_window = f"Deferred to Spare Parts Arrival Window (+{lead_time_hours/24.0:.1f} Days)"

        planned_end = planned_start + timedelta(hours=repair_hours)

        return {
            "target_window": target_window,
            "planned_start": planned_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "planned_end": planned_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "estimated_downtime_hours": repair_hours,
            "supply_constraint_alert": supply_constraint_alert,
            "operational_mitigation": operational_mitigation,
        }

    def _build_notification_payload(
        self,
        asset_id: str,
        fault_class: int,
        fault_name: str,
        anomaly_score: float,
        rul_hours: float,
        priority_code: str,
        priority_label: str,
        work_center: str,
        diagnosis_summary: str,
        repair_hours: float,
    ) -> Dict[str, Any]:
        """Construct compliant SAP S/4HANA PM Maintenance Notification payload."""
        now = datetime.now(timezone.utc)
        notif_type = "M1" if fault_class != FaultClass.HEALTHY else "M3"

        return {
            "NotificationType": notif_type,
            "NotificationText": f"AI Diagnostic Alert: {fault_name} on {asset_id}",
            "LongText": diagnosis_summary,
            "Equipment": asset_id,
            "FunctionalLocation": self.sap_config.functional_location,
            "Priority": priority_code,
            "PriorityText": priority_label,
            "ReportedByUser": "AGENTIC_SWARM_AI",
            "MainWorkCenter": work_center,
            "MaintenancePlant": self.sap_config.plant,
            "MalfunctionStartDate": now.strftime("%Y-%m-%d"),
            "MalfunctionStartTime": now.strftime("%H:%M:%S"),
            "BreakdownIndicator": priority_code == "1",
            "PredictedFaultClass": fault_class,
            "PredictedFailureMode": fault_name,
            "CompositeAnomalyScore": round(anomaly_score, 4),
            "EstimatedRULHours": round(rul_hours, 1),
            "RequiredLaborHours": round(repair_hours, 1),
        }

    def _build_work_order_payload(
        self,
        asset_id: str,
        fault_class: int,
        fault_name: str,
        priority_code: str,
        priority_label: str,
        work_center: str,
        scheduling: Dict[str, Any],
        repair_hours: float,
        required_parts: List[Dict[str, Any]],
        inventory_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct compliant SAP S/4HANA PM Maintenance Order payload (`API_MAINTENANCEORDER`)."""
        order_type = "PM01" if fault_class != FaultClass.HEALTHY else "PM02"

        # Standard 5-step maintenance operation breakdown
        operations = [
            {
                "Operation": "0010",
                "WorkCenter": work_center,
                "Plant": self.sap_config.plant,
                "OperationText": "Lockout/Tagout (LOTO) Electrical & Mechanical Isolation",
                "DurationHours": 0.5,
                "LaborUnit": "H",
            },
            {
                "Operation": "0020",
                "WorkCenter": work_center,
                "Plant": self.sap_config.plant,
                "OperationText": f"Disassembly and Inspection for {fault_name}",
                "DurationHours": round(max(0.5, repair_hours * 0.4), 1),
                "LaborUnit": "H",
            },
            {
                "Operation": "0030",
                "WorkCenter": work_center,
                "Plant": self.sap_config.plant,
                "OperationText": f"Component Replacement / Calibration Procedure",
                "DurationHours": round(max(0.5, repair_hours * 0.4), 1),
                "LaborUnit": "H",
            },
            {
                "Operation": "0040",
                "WorkCenter": work_center,
                "Plant": self.sap_config.plant,
                "OperationText": "Dynamic Verification, FFT Vibration & Temperature Re-test",
                "DurationHours": 0.5,
                "LaborUnit": "H",
            },
            {
                "Operation": "0050",
                "WorkCenter": work_center,
                "Plant": self.sap_config.plant,
                "OperationText": "Housekeeping, De-isolation, and Handover to Production",
                "DurationHours": 0.5,
                "LaborUnit": "H",
            },
        ]

        # Components / Material Reservations
        components = []
        for idx, part in enumerate(required_parts, start=1):
            mat_id = part.get("material_id", "")
            qty = part.get("quantity_required", 1)
            components.append({
                "ItemNumber": f"{idx * 10:04d}",
                "Material": mat_id,
                "RequirementQuantity": float(qty),
                "RequirementUnit": part.get("unit_of_measure", "EA"),
                "Plant": self.sap_config.plant,
                "StorageLocation": self.sap_config.storage_location,
                "IsAvailableInStock": inventory_status.get("parts_available", True),
            })

        return {
            "MaintenanceOrderType": order_type,
            "MaintenanceOrderDesc": f"AI PREDICTIVE: Repair {fault_name} on {asset_id}",
            "Equipment": asset_id,
            "FunctionalLocation": self.sap_config.functional_location,
            "MaintenancePlant": self.sap_config.plant,
            "MainWorkCenter": work_center,
            "PlannerGroup": self.sap_config.planner_group,
            "Priority": priority_code,
            "PriorityText": priority_label,
            "BasicStartDate": scheduling.get("planned_start", ""),
            "BasicEndDate": scheduling.get("planned_end", ""),
            "TargetSchedulingWindow": scheduling.get("target_window", ""),
            "TotalEstimatedDurationHours": round(repair_hours + 1.5, 1),
            "OperationalMitigation": scheduling.get("operational_mitigation"),
            "to_Operations": operations,
            "to_Components": components,
        }
