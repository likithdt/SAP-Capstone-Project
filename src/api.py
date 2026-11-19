"""
FastAPI REST API Service for Digital Twin, Agentic Swarm, and SAP Integration.
Provides endpoints for telemetry ingestion, Fiori card rendering, ERP actions, and jury demo.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import AssetConfig, DEFAULT_ASSET_CONFIG, FaultClass
from src.digital_twin.simulator import PhysicalSignalSimulator
from src.digital_twin.state_machine import DigitalTwin, AssetState
from src.digital_twin.fiori_adapter import FioriDashboardAdapter
from src.models.engine import PrognosticsEngine
from src.agents.swarm import DecisionSwarm
from src.sap.client import SAPClient
from src.cloud.lambda_handlers import ServerlessPipeline
from src.benchmarks.evaluator import KPIEvaluator
from src.benchmarks.fault_injector import FaultInjector
from src.benchmarks.jury_demo import JuryDemoRunner

app = FastAPI(
    title="Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine",
    description="Closed-Loop ERP Action with Multimodal Prognostics, LangGraph Swarm, and SAP S/4HANA OData Integration",
    version="1.0.0",
)

# Enable CORS for frontend dashboard (allowing cross-origin browser access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory for direct access at http://127.0.0.1:8000/dashboard
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="dashboard")
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")

# Shared system singleton state
twin = DigitalTwin()
engine = PrognosticsEngine()
swarm = DecisionSwarm()
sap_client = SAPClient()
pipeline = ServerlessPipeline(digital_twin=twin, prognostics_engine=engine, decision_swarm=swarm, sap_client=sap_client)
simulator = PhysicalSignalSimulator()
injector = FaultInjector()
evaluator = KPIEvaluator(engine=engine, swarm=swarm, sap_client=sap_client)
demo_runner = JuryDemoRunner()


class IngestTelemetryRequest(BaseModel):
    asset_id: str = "MOTOR-001"
    device_id: str = "ESP32-MOTOR-001"
    temperature_c: float = 49.5
    current_a: float = 1.22
    rms_vibration: float = 0.24
    speed_rpm: float = 1800.0
    vibration_x: List[float]


@app.get("/")
def get_root():
    """System health check and overview metadata."""
    return {
        "status": "ONLINE",
        "service": "Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine",
        "asset_id": twin.asset_id,
        "plant": twin.plant,
        "state": twin.state.value,
        "health_score": twin.health_score,
        "dashboard_url": "/dashboard",
        "endpoints": [
            "/dashboard",
            "/api/twin",
            "/api/fiori/overview",
            "/api/telemetry",
            "/api/sap/orders",
            "/api/sap/requisitions",
            "/api/demo/run",
            "/api/benchmarks/run",
        ],
    }


@app.get("/api/twin")
def get_digital_twin_state():
    """Retrieve current virtual representation of the rotating asset."""
    return twin.to_dict()


@app.get("/api/fiori/overview")
def get_fiori_overview():
    """Retrieve complete SAP Fiori / SAPUI5 Overview Page (OVP) card layout."""
    active_order = None
    if twin.active_work_orders:
        active_order = sap_client.get_order_status(twin.active_work_orders[0])

    return FioriDashboardAdapter.get_full_dashboard_snapshot(
        twin=twin,
        recent_order=active_order,
    )


@app.post("/api/telemetry")
def ingest_telemetry(req: IngestTelemetryRequest):
    """
    Ingest a telemetry window through the serverless pipeline:
    Validates, extracts features, runs prognostics, updates health score,
    and dispatches closed-loop SAP ERP actions if an anomaly is detected.
    """
    payload = req.model_dump()
    result = pipeline.execute_pipeline(payload)
    return result


@app.post("/api/telemetry/simulate")
def simulate_telemetry_batch(
    fault_class: int = Query(default=0, ge=0, le=6, description="Fault class 0-6 to simulate"),
    duration_sec: float = Query(default=0.5, ge=0.1, le=5.0),
):
    """Simulate a physics-based sensor burst and process it through the closed loop."""
    fc = FaultClass(fault_class)
    batch = simulator.generate_window(fc, duration_sec=duration_sec)
    payload = {
        "asset_id": batch.asset_id,
        "device_id": batch.device_id,
        "temperature_c": batch.mean_temperature,
        "current_a": batch.mean_current,
        "rms_vibration": batch.rms_vibration,
        "speed_rpm": batch.mean_speed_rpm,
        "vibration_x": batch.accel_x.tolist(),
        "vibration_y": batch.accel_y.tolist(),
        "vibration_z": batch.accel_z.tolist(),
    }
    return pipeline.execute_pipeline(payload)


@app.get("/api/sap/orders")
def get_sap_orders():
    """List all created SAP PM Maintenance Orders."""
    return {
        "total_orders": len(sap_client.ledger_orders),
        "orders": list(sap_client.ledger_orders.values()),
    }


@app.get("/api/sap/requisitions")
def get_sap_requisitions():
    """List all created SAP MM Purchase Requisitions."""
    return {
        "total_requisitions": len(sap_client.ledger_prs),
        "requisitions": list(sap_client.ledger_prs.values()),
    }


@app.post("/api/demo/run")
def run_live_jury_demo():
    """Run all 4 jury demonstration scenarios."""
    return demo_runner.run_all_scenarios()


@app.post("/api/benchmarks/run")
def run_kpi_benchmarks(samples_per_class: int = Query(default=2, ge=1, le=10)):
    """Run full KPI benchmark suite across all 7 fault classes."""
    dataset = injector.generate_benchmark_dataset(samples_per_class=samples_per_class, duration_sec=0.2)
    results = evaluator.evaluate_benchmark(dataset)
    return results.to_dict()


@app.get("/api/dataset/history")
def get_dataset_history(limit: int = Query(default=50, ge=5, le=500)):
    """Retrieve recent simulated dataset history for dashboard charting."""
    from pathlib import Path
    import pandas as pd
    csv_file = Path("data/simulated_telemetry_history.csv")
    if not csv_file.is_file():
        from scripts.generate_simulated_dataset import generate_full_dataset
        generate_full_dataset()
    df = pd.read_csv(csv_file)
    records = df.tail(limit).to_dict(orient="records")
    return {
        "total_records": len(df),
        "limit": limit,
        "records": records,
    }


@app.post("/api/dataset/reset")
def reset_system_state():
    """Reset Digital Twin and SAP client to pristine healthy state for demonstrations."""
    twin.reset()
    sap_client.ledger_orders.clear()
    sap_client.ledger_prs.clear()
    return {
        "status": "RESET",
        "asset_id": twin.asset_id,
        "state": twin.state.value,
        "health_score": twin.health_score,
        "active_work_orders": twin.active_work_orders,
        "message": "System digital twin and ERP ledgers reset to nominal baseline.",
    }

