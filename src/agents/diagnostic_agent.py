"""
Diagnostic Agent with Vector-Search RAG over Equipment Maintenance Manuals.
Analyzes prognostic anomalies, performs semantic retrieval over SOP manuals,
identifies mechanical root causes, and specifies required replacement spare parts.
"""

from typing import Any, Dict, List, Optional
from src.config import AgentConfig, DEFAULT_AGENT_CONFIG, FaultClass
from src.agents.rag.vector_store import MaintenanceVectorStore
from src.agents.rag.knowledge_base import EQUIPMENT_MAINTENANCE_MANUALS


class DiagnosticAgent:
    """
    RAG-powered Diagnostic Agent.
    Interprets prognostics anomaly telemetry, queries engineering manuals via vector similarity,
    and produces grounded root-cause diagnoses with exact SAP spare parts lists.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        vector_store: Optional[MaintenanceVectorStore] = None,
    ):
        self.config = config or DEFAULT_AGENT_CONFIG
        self.vector_store = vector_store or MaintenanceVectorStore(self.config)
        self._manuals_by_class = {m.fault_class: m for m in EQUIPMENT_MAINTENANCE_MANUALS}

    def diagnose(
        self,
        fault_class: int,
        fault_name: str,
        anomaly_score: float,
        risk_level: str,
        confidence: float,
        rul_hours: float,
        evidence: Optional[List[str]] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute RAG diagnostic analysis over equipment maintenance manuals.

        Args:
            fault_class: Target class 0-6 from prognostics engine
            fault_name: Name of predicted failure mode
            anomaly_score: Composite anomaly score in [0, 1]
            risk_level: LOW, MEDIUM, HIGH, CRITICAL
            confidence: Classifier confidence
            rul_hours: Remaining useful life in operating hours
            evidence: Root cause evidence bullets from prognostics engine
            features: Extracted sensor signal features

        Returns:
            Dict containing RAG context, diagnosis summary, required spare parts, and repair hours.
        """
        evidence_list = evidence or []
        feats = features or {}

        # 1. Formulate rich semantic query for vector search
        evidence_text = " ".join(evidence_list)
        query = f"{fault_name} {evidence_text}".strip()
        if not query:
            query = f"fault class {fault_class} standard operating procedure"

        # 2. Query Vector Store for relevant SOPs
        retrieved_docs = self.vector_store.search(
            query=query,
            n_results=self.config.rag_top_k,
        )

        # 3. Retrieve ground-truth manual matching the fault class
        matched_manual = self._manuals_by_class.get(fault_class)
        required_parts = [p.copy() for p in matched_manual.recommended_parts] if matched_manual else []
        estimated_hours = matched_manual.estimated_hours if matched_manual else 1.0
        sop_code = matched_manual.sop_code if matched_manual else "SOP-GEN-001"

        # 4. Synthesize explainable diagnostic summary
        diagnosis_summary = self._synthesize_diagnosis(
            fault_class=fault_class,
            fault_name=fault_name,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            confidence=confidence,
            rul_hours=rul_hours,
            evidence=evidence_list,
            retrieved_docs=retrieved_docs,
            sop_code=sop_code,
            estimated_hours=estimated_hours,
            parts=required_parts,
        )

        recommended_action = (
            f"Execute {sop_code} ({fault_name}) within {rul_hours:.1f} operating hours. "
            f"Estimated work duration: {estimated_hours:.1f} hours."
        ) if fault_class != 0 else "Normal operation. No maintenance action required."

        return {
            "rag_context": retrieved_docs,
            "diagnosis_summary": diagnosis_summary,
            "recommended_action": recommended_action,
            "required_spare_parts": required_parts,
            "estimated_repair_hours": estimated_hours,
            "sop_code": sop_code,
        }

    def _synthesize_diagnosis(
        self,
        fault_class: int,
        fault_name: str,
        anomaly_score: float,
        risk_level: str,
        confidence: float,
        rul_hours: float,
        evidence: List[str],
        retrieved_docs: List[Dict[str, Any]],
        sop_code: str,
        estimated_hours: float,
        parts: List[Dict[str, Any]],
    ) -> str:
        """Construct grounded, explainable engineering diagnostic rationale."""
        if fault_class == FaultClass.HEALTHY:
            return (
                "ASSET HEALTH STATUS: OPTIMAL (Class 0 - Healthy).\n"
                "All vibration, thermal, and current metrics are within nominal limits. "
                "No mechanical or electrical degradation detected."
            )

        top_manual_title = retrieved_docs[0]["title"] if retrieved_docs else "Equipment Maintenance Manual"
        top_sop = retrieved_docs[0].get("sop_code", sop_code) if retrieved_docs else sop_code

        parts_summary = ", ".join([f"{p['material_name']} (Qty: {p['quantity_required']})" for p in parts]) or "None"
        evidence_summary = "\n  - ".join(evidence) if evidence else "Elevated vibration anomalies detected."

        summary = (
            f"DIAGNOSTIC ROOT-CAUSE SUMMARY:\n"
            f"• Detected Condition: {fault_name} (Class {fault_class})\n"
            f"• Anomaly Score: {anomaly_score:.2f} | Risk Severity: {risk_level} | Classifier Confidence: {confidence*100:.1f}%\n"
            f"• Remaining Useful Life (RUL): {rul_hours:.1f} hours before projected functional failure\n"
            f"• Grounded Manual Reference: {top_sop} - {top_manual_title}\n"
            f"• Physical Evidence:\n  - {evidence_summary}\n"
            f"• Required Spare Parts: {parts_summary}\n"
            f"• Estimated Labor Duration: {estimated_hours:.1f} hours"
        )
        return summary
