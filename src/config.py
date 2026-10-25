"""
Project Configuration and Physical Asset Specifications.
Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine with Closed-Loop ERP Action.
"""

from enum import IntEnum
from typing import Dict, Tuple, Any
from pydantic import BaseModel, Field


class FaultClass(IntEnum):
    """Fault class categorization matching physical modeling specifications and 1D-CNN targets."""
    HEALTHY = 0
    BEARING_WEAR = 1
    ROTOR_IMBALANCE = 2
    SHAFT_MISALIGNMENT = 3
    THERMAL_OVERHEATING = 4
    MOTOR_OVERLOAD = 5
    SENSOR_ANOMALY = 6

    @property
    def label(self) -> str:
        labels = {
            FaultClass.HEALTHY: "Healthy",
            FaultClass.BEARING_WEAR: "Bearing Wear (BPFO)",
            FaultClass.ROTOR_IMBALANCE: "Rotor Imbalance (1X)",
            FaultClass.SHAFT_MISALIGNMENT: "Shaft Misalignment (2X/3X)",
            FaultClass.THERMAL_OVERHEATING: "Thermal Overheating",
            FaultClass.MOTOR_OVERLOAD: "Motor Overload",
            FaultClass.SENSOR_ANOMALY: "Sensor Anomaly",
        }
        return labels[self]


class AssetConfig(BaseModel):
    """Nominal physical specifications of rotating industrial asset."""
    asset_id: str = "MOTOR-001"
    device_id: str = "ESP32-MOTOR-001"
    asset_name: str = "Conveyor Drive Motor"
    plant: str = "PLANT-01"
    nominal_speed_rpm: float = Field(default=1800.0, description="Nominal rotational speed N (RPM)")
    sampling_rate_hz: float = Field(default=1000.0, description="Sensor sampling rate fs (Hz)")
    line_frequency_hz: float = Field(default=50.0, description="Grid electrical supply frequency (Hz)")
    bpfo_factor: float = Field(default=3.58, description="BPFO fault frequency multiplier (3.58 * f0)")

    @property
    def fundamental_freq_hz(self) -> float:
        """Fundamental rotational frequency f0 (Hz) = N / 60."""
        return self.nominal_speed_rpm / 60.0

    @property
    def bpfo_freq_hz(self) -> float:
        """Ball Pass Frequency Outer Raceway (BPFO) in Hz."""
        return self.bpfo_factor * self.fundamental_freq_hz

    @property
    def nyquist_freq_hz(self) -> float:
        """Nyquist frequency (fs / 2) in Hz."""
        return self.sampling_rate_hz / 2.0


class ClassThresholds:
    """Quantitative physical parameter targets for each fault class."""
    
    # Class 0: Healthy
    HEALTHY_RMS_MIN_G: float = 0.15
    HEALTHY_RMS_MAX_G: float = 0.35
    HEALTHY_TEMP_MIN_C: float = 45.0
    HEALTHY_TEMP_MAX_C: float = 52.0
    HEALTHY_CURRENT_MIN_A: float = 1.1
    HEALTHY_CURRENT_MAX_A: float = 1.3

    # Class 1: Bearing Wear
    BEARING_KURTOSIS_THRESHOLD: float = 4.5
    BEARING_CREST_FACTOR_MIN: float = 4.0

    # Class 2: Rotor Imbalance
    IMBALANCE_RMS_MIN_G: float = 0.8
    IMBALANCE_1X_POWER_RATIO_MIN: float = 0.70

    # Class 3: Shaft Misalignment
    MISALIGNMENT_2X_HARMONIC_MIN_RATIO: float = 0.50
    MISALIGNMENT_AXIAL_RATIO_MIN: float = 0.60

    # Class 4: Thermal Overheating
    THERMAL_OVERHEATING_START_C: float = 55.0
    THERMAL_OVERHEATING_CRITICAL_C: float = 85.0

    # Class 5: Motor Overload
    OVERLOAD_CURRENT_MIN_A: float = 2.4
    OVERLOAD_CURRENT_MAX_A: float = 3.8
    OVERLOAD_LINE_RIPPLE_FREQ_HZ: float = 100.0  # 2 * 50 Hz

    # Class 6: Sensor Anomaly
    SENSOR_SPIKE_THRESHOLD_G: float = 15.0


# Default asset configuration instance
DEFAULT_ASSET_CONFIG = AssetConfig()


class MLOpsConfig(BaseModel):
    """Configuration for MLflow experiment tracking and Evidently AI drift monitoring."""
    experiment_name: str = "Predictive-Maintenance-Prognostics"
    tracking_uri: str = "sqlite:///data/mlflow.db"
    drift_share_threshold: float = Field(default=0.30, description="Threshold ratio of drifting features to trigger alarm")
    drift_p_value_threshold: float = Field(default=0.05, description="Significance level for drift statistical hypothesis tests")
    report_output_dir: str = "data/monitoring/reports"
    model_registry_name: str = "ConveyorMotorPrognosticsEngine"


DEFAULT_MLOPS_CONFIG = MLOpsConfig()


class SAPConfig(BaseModel):
    """Configuration for SAP S/4HANA PM and MM enterprise integration."""
    plant: str = "PLANT-01"
    maintenance_plant: str = "PLANT-01"
    storage_location: str = "SL01"
    purchasing_group: str = "001"
    purchasing_organization: str = "1000"
    cost_center: str = "CC-MAINT-01"
    work_center_mechanical: str = "MECH-01"
    work_center_electrical: str = "ELEC-01"
    planner_group: str = "PG-01"
    functional_location: str = "PLANT01-CONV-DRV"
    odata_pm_base_url: str = "https://sandbox.sap.s4hana.ondemand.com/sap/opu/odata4/sap/api_maintenanceorder/srvd_a2x/sap/maintenanceorder/0001"
    odata_mm_stock_base_url: str = "https://sandbox.sap.s4hana.ondemand.com/sap/opu/odata4/sap/api_product_srv/srvd_a2x/sap/product/0001"
    odata_mm_pr_base_url: str = "https://sandbox.sap.s4hana.ondemand.com/sap/opu/odata4/sap/api_purchaserequisition_process_srv/srvd_a2x/sap/purchaserequisition/0001"
    auth_type: str = "OAuth2"


DEFAULT_SAP_CONFIG = SAPConfig()


class AgentConfig(BaseModel):
    """Configuration for LangGraph Agentic AI Decision Swarm and RAG Vector Store."""
    chroma_persist_dir: str = "data/chroma"
    collection_name: str = "equipment_maintenance_manuals"
    rag_top_k: int = Field(default=3, description="Number of maintenance manual chunks to retrieve")
    llm_model: str = "gpt-4o-mini"
    enable_llm: bool = Field(default=False, description="When False, uses deterministic expert RAG synthesizer")
    shift_duration_hours: float = 8.0
    emergency_rul_threshold_hours: float = 12.0
    high_priority_rul_threshold_hours: float = 48.0
    medium_priority_rul_threshold_hours: float = 100.0


DEFAULT_AGENT_CONFIG = AgentConfig()


