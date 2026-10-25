"""
SAP Fiori / SAPUI5 Dashboard Integration Adapter.
Serializes Digital Twin virtual model state, prognostics telemetry, and SAP ERP actions
into standardized SAP Fiori Overview Page (OVP) card payloads.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from src.digital_twin.state_machine import DigitalTwin, AssetState


class FioriDashboardAdapter:
    """
    Adapter for SAP Fiori / SAPUI5 Enterprise Frontends.
    Transforms live Digital Twin state and ERP records into Fiori Overview Card schemas.
    """

    @classmethod
    def get_health_gauge_card(cls, twin: DigitalTwin) -> Dict[str, Any]:
        """SAP Fiori Analytical / Gauge Card for Asset Health Score."""
        breakdown = twin.last_health_breakdown
        return {
            "card_type": "AnalyticalCard",
            "card_id": "fiori_card_asset_health_gauge",
            "title": f"Asset Health Score — {twin.asset_id}",
            "sub_title": f"Plant: {twin.plant} | Equipment: {twin.sap_equipment_number}",
            "value": twin.health_score,
            "unit": "/ 100",
            "state": twin.state.value,
            "color_code": twin.state.color_code,
            "criticality": "Positive" if twin.health_score >= 80 else ("Critical" if twin.health_score < 45 else "Warning"),
            "target_value": 100.0,
            "thresholds": {
                "critical": 45.0,
                "warning": 65.0,
                "watchlist": 80.0,
                "healthy": 100.0,
            },
            "breakdown": {
                "anomaly_penalty": breakdown.anomaly_penalty if breakdown else 0.0,
                "temperature_penalty": breakdown.temperature_penalty if breakdown else 0.0,
                "current_penalty": breakdown.current_penalty if breakdown else 0.0,
                "rul_penalty": breakdown.rul_penalty if breakdown else 0.0,
            },
        }

    @classmethod
    def get_telemetry_card(cls, twin: DigitalTwin) -> Dict[str, Any]:
        """SAP Fiori Key Performance Indicator (KPI) Metric Card."""
        return {
            "card_type": "NumericCard",
            "card_id": "fiori_card_live_telemetry",
            "title": "Live Sensor Telemetry",
            "sub_title": "Vibration, Temperature, Current, Speed",
            "metrics": [
                {
                    "label": "Radial Vibration RMS",
                    "value": twin.rms_vibration,
                    "unit": "g",
                    "status": "Error" if twin.rms_vibration > 0.40 else ("Warning" if twin.rms_vibration > 0.35 else "Success"),
                    "threshold": "Nominal: 0.15 - 0.35 g",
                },
                {
                    "label": "Stator Temperature",
                    "value": twin.temperature_c,
                    "unit": "°C",
                    "status": "Error" if twin.temperature_c > 85.0 else ("Warning" if twin.temperature_c > 55.0 else "Success"),
                    "threshold": "Critical: > 85.0 °C",
                },
                {
                    "label": "Motor Current Draw",
                    "value": twin.current_a,
                    "unit": "A",
                    "status": "Error" if twin.current_a > 2.40 else ("Warning" if twin.current_a > 1.30 else "Success"),
                    "threshold": "Nominal: 1.10 - 1.30 A",
                },
                {
                    "label": "Shaft Rotational Speed",
                    "value": twin.speed_rpm,
                    "unit": "RPM",
                    "status": "Success",
                    "threshold": "Nominal: 1800 RPM",
                },
            ],
        }

    @classmethod
    def get_prognostics_card(cls, twin: DigitalTwin) -> Dict[str, Any]:
        """SAP Fiori Machine Learning Prediction Card."""
        report = twin.last_diagnostic_report
        return {
            "card_type": "ListCard",
            "card_id": "fiori_card_prognostics_prediction",
            "title": "Deep Learning Multimodal Prognostics",
            "sub_title": "AI Diagnosis & Remaining Useful Life (RUL)",
            "predicted_failure_mode": twin.predicted_failure_mode,
            "predicted_class": report.predicted_class if report else 0,
            "confidence_percent": round(report.confidence * 100.0, 1) if report else 95.0,
            "anomaly_score": twin.anomaly_score,
            "risk_level": twin.risk_level,
            "rul_hours": twin.rul_hours,
            "rul_status": "Urgent" if twin.rul_hours < 12.0 else ("Warning" if twin.rul_hours < 48.0 else "Normal"),
            "evidence": list(report.evidence) if report else ["Nominal baseline operation"],
            "recommended_action": report.recommended_action if report else "Continuous monitoring active.",
        }

    @classmethod
    def get_erp_actions_card(
        cls,
        twin: DigitalTwin,
        recent_order: Optional[Dict[str, Any]] = None,
        recent_pr: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """SAP Fiori Enterprise Work Order and Procurement Status Card."""
        active_wo_id = twin.active_work_orders[0] if twin.active_work_orders else None
        active_pr_id = twin.active_purchase_requisitions[0] if twin.active_purchase_requisitions else None

        return {
            "card_type": "TableCard",
            "card_id": "fiori_card_erp_actions",
            "title": "Closed-Loop SAP ERP Actions",
            "sub_title": "SAP PM Work Orders & SAP MM Requisitions",
            "active_work_order_id": active_wo_id,
            "active_pr_id": active_pr_id,
            "order_details": recent_order,
            "pr_details": recent_pr,
            "status_banner": (
                f"ACTIVE MAINTENANCE: Work Order #{active_wo_id} in execution"
                if active_wo_id
                else "No active maintenance work orders pending."
            ),
        }

    @classmethod
    def get_full_dashboard_snapshot(
        cls,
        twin: DigitalTwin,
        recent_order: Optional[Dict[str, Any]] = None,
        recent_pr: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble unified JSON payload representing complete SAP Fiori Launchpad screen."""
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "timestamp": now_ts,
            "system": "SAP S/4HANA Cloud Predictive Maintenance",
            "asset_overview": twin.to_dict(),
            "fiori_cards": [
                cls.get_health_gauge_card(twin),
                cls.get_telemetry_card(twin),
                cls.get_prognostics_card(twin),
                cls.get_erp_actions_card(twin, recent_order, recent_pr),
            ],
            "state_transitions": [
                {
                    "timestamp": h.timestamp,
                    "from_state": h.previous_state.value,
                    "to_state": h.new_state.value,
                    "health_score": h.health_score,
                    "reason": h.trigger_reason,
                }
                for h in twin.state_history[-5:]
            ],
        }
