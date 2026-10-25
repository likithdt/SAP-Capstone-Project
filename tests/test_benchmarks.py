"""
Unit and Integration Tests for Step 8: Fault Injection Benchmarking, KPI Evaluation & Jury Demo.
Validates:
  - Systematic physical fault injection across all 7 operational failure classes
  - System KPI Benchmarking (Accuracy, Confusion Matrix, RUL RMSE, End-to-End Latency < 5s)
  - Capstone Targets: False Positive Rate < 5%, SAP Work Order Success Rate > 95%
  - Live Jury Demonstration Runner executing all 4 real-world operational scenarios
"""

import os
import pytest
from src.config import FaultClass
from src.benchmarks.fault_injector import FaultInjector
from src.benchmarks.evaluator import KPIEvaluator, BenchmarkKPIResults
from src.benchmarks.jury_demo import JuryDemoRunner


@pytest.fixture
def fault_injector():
    return FaultInjector()


@pytest.fixture
def kpi_evaluator():
    return KPIEvaluator()


@pytest.fixture
def jury_demo():
    return JuryDemoRunner()


# -----------------------------------------------------------------------------
# 1. Fault Injection Tests
# -----------------------------------------------------------------------------
def test_fault_injector_all_classes(fault_injector):
    """Test fault generation across all 7 operational classes."""
    for fc in FaultClass:
        batch = fault_injector.inject_fault(fc, duration_sec=0.2, seed=int(fc))
        assert batch.fault_class == int(fc)
        assert len(batch.accel_x) == 200
        assert batch.sampling_rate_hz == 1000.0


def test_fault_injector_degradation_trajectory(fault_injector):
    """Test progressive multi-stage degradation sequence."""
    trajectory = fault_injector.generate_degradation_trajectory(
        target_fault=FaultClass.BEARING_WEAR,
        num_steps=4,
        duration_sec=0.2,
    )
    assert len(trajectory) == 4
    # First stage must be healthy
    assert trajectory[0].fault_class == FaultClass.HEALTHY
    # Subsequent stages must reflect target fault
    for b in trajectory[1:]:
        assert b.fault_class == FaultClass.BEARING_WEAR


# -----------------------------------------------------------------------------
# 2. KPI Evaluation & Benchmark Tests (Section 19)
# -----------------------------------------------------------------------------
def test_kpi_evaluator_benchmark_execution(fault_injector, kpi_evaluator, tmp_path):
    """
    Execute full KPI benchmark across multi-class dataset and evaluate against Capstone targets:
      - Model Accuracy > 85%
      - False Positive Rate < 5%
      - End-to-end alert latency < 5.0 seconds
      - SAP Work Order success rate > 95%
    """
    # 2 samples per class = 14 evaluation windows
    dataset = fault_injector.generate_benchmark_dataset(samples_per_class=2, duration_sec=0.2, base_seed=42)
    assert len(dataset) == 14

    out_dir = str(tmp_path / "benchmarks")
    results: BenchmarkKPIResults = kpi_evaluator.evaluate_benchmark(dataset, output_dir=out_dir)

    # 1. Model Quality Assertions
    assert results.accuracy >= 0.85, f"Expected accuracy >= 0.85, got {results.accuracy:.2f}"
    assert results.macro_f1 >= 0.80, f"Expected macro F1 >= 0.80, got {results.macro_f1:.2f}"
    assert results.false_positive_rate <= 0.05, f"Expected FPR <= 0.05, got {results.false_positive_rate:.2f}"
    assert len(results.confusion_matrix) == 7

    # 2. RUL Regression Assertions
    assert results.rul_mae_hours >= 0.0
    assert results.rul_rmse_hours >= 0.0

    # 3. Latency Assertions (Target: < 5-10 seconds per Section 19)
    assert results.total_end_to_end_latency_sec < 5.0, f"Expected latency < 5.0s, got {results.total_end_to_end_latency_sec:.3f}s"
    assert results.inference_latency_sec < 1.0

    # 4. ERP Action Assertions (Target: > 95% per Section 19)
    assert results.sap_work_order_success_rate >= 0.95, f"Expected SAP success rate >= 0.95, got {results.sap_work_order_success_rate:.2f}"
    assert results.successful_orders_created > 0

    # 5. JSON Artifact Verification
    artifact_path = os.path.join(out_dir, "benchmark_kpi_results.json")
    assert os.path.exists(artifact_path)


# -----------------------------------------------------------------------------
# 3. Live Jury Demo Scenarios
# -----------------------------------------------------------------------------
def test_jury_demo_all_scenarios(jury_demo):
    """Test automated jury presentation suite executing all 4 operational scenarios."""
    summary = jury_demo.run_all_scenarios()
    assert "scenario_1_healthy" in summary
    assert "scenario_2_bearing_defect" in summary
    assert "scenario_3_thermal_stockout" in summary
    assert "scenario_4_sensor_glitch" in summary

    # Scenario 1 assertions
    s1 = summary["scenario_1_healthy"]
    assert s1["final_state"] == "HEALTHY"
    assert s1["work_order_created"] is False
    assert s1["health_score"] >= 80.0

    # Scenario 2 assertions
    s2 = summary["scenario_2_bearing_defect"]
    assert s2["final_state"] == "MAINTENANCE"
    assert s2["sap_work_order_id"] is not None
    assert s2["sap_order_type"] == "PM01"
    assert s2["parts_available"] is True

    # Scenario 3 assertions
    s3 = summary["scenario_3_thermal_stockout"]
    assert s3["final_state"] == "MAINTENANCE"
    assert s3["sap_mm_pr_id"] is not None
    assert s3["priority"] == "1"
    assert s3["supply_alert"] is True

    # Scenario 4 assertions
    s4 = summary["scenario_4_sensor_glitch"]
    assert s4["fault_class"] == FaultClass.SENSOR_ANOMALY
    assert s4["bearing_replacement_avoided"] is True
