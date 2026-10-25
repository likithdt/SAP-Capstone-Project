"""
Unit and Integration Tests for FastAPI Digital Twin & ERP REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_api_root():
    """Test root health check and metadata endpoint."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ONLINE"
    assert "asset_id" in data
    assert "endpoints" in data


def test_api_get_twin():
    """Test retrieving current Digital Twin virtual model."""
    resp = client.get("/api/twin")
    assert resp.status_code == 200
    twin = resp.json()
    assert twin["asset_id"] == "MOTOR-001"
    assert "health_score" in twin
    assert "state" in twin


def test_api_get_fiori_overview():
    """Test retrieving SAP Fiori Overview Page layout."""
    resp = client.get("/api/fiori/overview")
    assert resp.status_code == 200
    ovp = resp.json()
    assert "fiori_cards" in ovp
    assert len(ovp["fiori_cards"]) == 4
    card_types = [c["card_type"] for c in ovp["fiori_cards"]]
    assert "AnalyticalCard" in card_types
    assert "NumericCard" in card_types


def test_api_simulate_telemetry():
    """Test simulating physics-based telemetry burst and processing via pipeline."""
    resp = client.post("/api/telemetry/simulate?fault_class=1&duration_sec=0.2")
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "COMPLETED"
    assert "ingest" in res
    assert "inference" in res
    assert res["sap_dispatch"]["dispatched"] is True


def test_api_sap_orders_and_requisitions():
    """Test querying created SAP PM Work Orders and SAP MM Purchase Requisitions."""
    resp_orders = client.get("/api/sap/orders")
    assert resp_orders.status_code == 200
    orders = resp_orders.json()
    assert "total_orders" in orders

    resp_prs = client.get("/api/sap/requisitions")
    assert resp_prs.status_code == 200
    prs = resp_prs.json()
    assert "total_requisitions" in prs


def test_api_demo_run():
    """Test running live jury demo scenarios via API."""
    resp = client.post("/api/demo/run")
    assert resp.status_code == 200
    demo = resp.json()
    assert "scenario_1_healthy" in demo
    assert "scenario_2_bearing_defect" in demo
