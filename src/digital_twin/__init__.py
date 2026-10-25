"""
Digital Twin module for asset simulation, state machine transitions, and Fiori UI modeling.
"""

from src.digital_twin.simulator import PhysicalSignalSimulator, TelemetryBatch
from src.digital_twin.state_machine import (
    AssetState,
    HealthScoreWeights,
    HealthScoreBreakdown,
    HealthScoreEngine,
    DigitalTwin,
    DigitalTwinStateHistory,
)
from src.digital_twin.fiori_adapter import FioriDashboardAdapter
from src.config import AssetConfig, FaultClass, ClassThresholds

__all__ = [
    "PhysicalSignalSimulator",
    "TelemetryBatch",
    "AssetState",
    "HealthScoreWeights",
    "HealthScoreBreakdown",
    "HealthScoreEngine",
    "DigitalTwin",
    "DigitalTwinStateHistory",
    "FioriDashboardAdapter",
    "AssetConfig",
    "FaultClass",
    "ClassThresholds",
]
