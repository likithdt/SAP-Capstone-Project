"""
Systematic Fault Injection Engine for Industrial Rotating Asset.
Injects controlled physics-based faults across all 7 operational failure classes:
  - Class 0: Healthy
  - Class 1: Bearing Wear (BPFO)
  - Class 2: Rotor Imbalance (1X)
  - Class 3: Shaft Misalignment (2X/3X)
  - Class 4: Thermal Overheating
  - Class 5: Motor Overload
  - Class 6: Sensor Glitch / Anomaly
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import AssetConfig, DEFAULT_ASSET_CONFIG, FaultClass
from src.digital_twin.simulator import PhysicalSignalSimulator, TelemetryBatch


class FaultInjector:
    """
    Simulates operational degradation and abrupt fault injection for testing and benchmarks.
    """

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        simulator: Optional[PhysicalSignalSimulator] = None,
    ):
        self.config = config or DEFAULT_ASSET_CONFIG
        self.simulator = simulator or PhysicalSignalSimulator(self.config)

    def inject_fault(
        self,
        fault_class: FaultClass,
        duration_sec: float = 1.0,
        seed: Optional[int] = None,
    ) -> TelemetryBatch:
        """
        Generate a synchronized sensor burst containing the specified physical defect.
        """
        return self.simulator.generate_window(
            fault_class=fault_class,
            duration_sec=duration_sec,
            seed=seed,
        )

    def generate_benchmark_dataset(
        self,
        samples_per_class: int = 10,
        duration_sec: float = 0.5,
        base_seed: int = 1000,
    ) -> List[Tuple[TelemetryBatch, FaultClass]]:
        """
        Generate a multi-class evaluation dataset across all 7 operational classes.
        Returns a list of (TelemetryBatch, ground_truth_fault_class) tuples.
        """
        dataset = []
        for fc in FaultClass:
            for i in range(samples_per_class):
                seed = base_seed + int(fc) * 100 + i
                batch = self.simulator.generate_window(
                    fault_class=fc,
                    duration_sec=duration_sec,
                    seed=seed,
                )
                dataset.append((batch, fc))
        return dataset

    def generate_degradation_trajectory(
        self,
        target_fault: FaultClass,
        num_steps: int = 6,
        duration_sec: float = 0.5,
    ) -> List[TelemetryBatch]:
        """
        Generate a progressive multi-stage degradation sequence from nominal healthy to severe defect.
        """
        trajectory = []
        # Stage 0: Healthy
        trajectory.append(self.simulator.generate_window(FaultClass.HEALTHY, duration_sec=duration_sec, seed=42))

        # Intermediate stages
        for step in range(1, num_steps):
            seed = 500 + step
            batch = self.simulator.generate_window(target_fault, duration_sec=duration_sec, seed=seed)
            trajectory.append(batch)

        return trajectory
