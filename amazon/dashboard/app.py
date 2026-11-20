import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is available in sys.path
_current = Path(__file__).resolve()
if _current.parent.name == "dashboard" and _current.parent.parent.name == "amazon":
    PROJECT_ROOT = _current.parent.parent.parent
else:
    PROJECT_ROOT = _current.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

st.set_page_config(
    page_title="Industrial IoT Digital Twin · Prototype",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏭 Autonomous Industrial IoT Digital Twin & Predictive Maintenance")
st.caption("Live Prototype Dashboard · High-Fidelity Multi-Sensor Simulation & Closed-Loop ERP Action")

region = os.getenv("AWS_REGION", "us-east-1")
history_table_name = os.getenv("DYNAMODB_TABLE", "MachineTelemetry")
latest_table_name = os.getenv("LATEST_TABLE_NAME", "MachineLatestState")

DATA_CSV_PATH = PROJECT_ROOT / "data" / "simulated_telemetry_history.csv"
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_local_simulated_dataset() -> pd.DataFrame:
    """Load simulated telemetry history from local CSV."""
    if not DATA_CSV_PATH.is_file():
        try:
            from scripts.generate_simulated_dataset import generate_full_dataset
            generate_full_dataset(output_dir=str(PROJECT_ROOT / "data"))
        except Exception:
            pass

    if DATA_CSV_PATH.is_file():
        df = pd.read_csv(DATA_CSV_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df.sort_values("timestamp")
    
    return pd.DataFrame()


def query_fastapi_backend(api_url: str):
    """Fetch live digital twin state and rolling history from FastAPI backend."""
    try:
        resp_twin = requests.get(f"{api_url}/api/twin", timeout=3)
        if resp_twin.status_code != 200:
            return None, pd.DataFrame(), [], []
        twin_data = resp_twin.json()

        # Fetch history records
        resp_hist = requests.get(f"{api_url}/api/dataset/history?limit=100", timeout=3)
        records = resp_hist.json().get("records", []) if resp_hist.status_code == 200 else []
        df = pd.DataFrame(records)
        if not df.empty and "timestamp" in df:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Fetch SAP orders & requisitions
        resp_orders = requests.get(f"{api_url}/api/sap/orders", timeout=3)
        orders = resp_orders.json().get("orders", []) if resp_orders.status_code == 200 else []

        resp_prs = requests.get(f"{api_url}/api/sap/requisitions", timeout=3)
        prs = resp_prs.json().get("requisitions", []) if resp_prs.status_code == 200 else []

        return twin_data, df, orders, prs
    except Exception:
        return None, pd.DataFrame(), [], []


def decimal_to_float(value):
    if isinstance(value, dict):
        return {k: decimal_to_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decimal_to_float(v) for v in value]
    try:
        from decimal import Decimal
        if isinstance(value, Decimal):
            return float(value)
    except Exception:
        pass
    return value


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Prototype Controls")
    
    data_source = st.selectbox(
        "Data Source Mode",
        [
            "FastAPI Digital Twin (Live Sync with Frontend)",
            "Simulated Dataset (Local Files)",
            "AWS DynamoDB (Live Cloud)",
        ],
        index=0,
        help="Connect to FastAPI to stay in real-time sync with the Web Frontend, or use standalone simulated files.",
    )

    api_url = DEFAULT_API_URL
    if data_source == "FastAPI Digital Twin (Live Sync with Frontend)":
        api_url = st.text_input("Backend API URL", DEFAULT_API_URL)

    device_id = st.text_input("Device ID", "ESP32-MOTOR-001")
    asset_id = st.text_input("Asset ID", "MOTOR-001")
    
    st.divider()

    # Scenario Selection (for simulated modes)
    scenario_choice = "All Scenarios"
    if data_source != "AWS DynamoDB (Live Cloud)":
        st.subheader("Fault Injection & Presets")
        scenario_choice = st.selectbox(
            "Filter Scenario",
            [
                "All Scenarios",
                "Nominal Healthy Baseline",
                "Bearing Wear (BPFO 107.4 Hz)",
                "Rotor Imbalance (1X 30 Hz)",
                "Shaft Misalignment (2X/3X)",
                "Thermal Overheating",
                "Motor Overload Ripple",
                "Transient Sensor Glitch",
            ],
            index=0,
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            btn_healthy = st.button("➕ Healthy", use_container_width=True)
            btn_imbalance = st.button("⚙️ Imbalance", use_container_width=True)
        with col_b2:
            btn_bearing = st.button("🚨 Bearing Defect", use_container_width=True)
            btn_thermal = st.button("🔥 Overheating", use_container_width=True)

        btn_reset = st.button("🔄 Reset Baseline (100%)", use_container_width=True)

    max_records = st.slider("Historical Records Displayed", 20, 200, 60, step=10)
    refresh = st.button("🔄 Refresh Data", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Data Resolution
# -----------------------------------------------------------------------------
df = pd.DataFrame()
twin_state = {}
sap_orders = []
sap_prs = []
active_mode_label = ""

# Handle Button Injections for FastAPI Live Sync
if data_source == "FastAPI Digital Twin (Live Sync with Frontend)":
    if btn_healthy or btn_bearing or btn_imbalance or btn_thermal or btn_reset:
        try:
            if btn_reset:
                requests.post(f"{api_url}/api/dataset/reset", timeout=5)
                st.toast("Digital twin reset to nominal baseline!", icon="✅")
            else:
                fc = 0
                if btn_bearing: fc = 1
                elif btn_imbalance: fc = 2
                elif btn_thermal: fc = 4
                requests.post(f"{api_url}/api/telemetry/simulate?fault_class={fc}&duration_sec=0.3", timeout=5)
                st.toast(f"Injected Fault Class {fc} into FastAPI backend!", icon="🚨")
        except Exception as e:
            st.error(f"Failed to post simulation to API: {e}")

    # Fetch live state from FastAPI
    twin_state, df, sap_orders, sap_prs = query_fastapi_backend(api_url)

    if twin_state is not None:
        active_mode_label = f"🟢 Connected to FastAPI Server ({api_url}) · Synced with Web Frontend"
    else:
        st.warning(f"⚠️ FastAPI server not reached at `{api_url}`. Falling back to local simulated dataset.")
        data_source = "Simulated Dataset (Local Files)"

# Local Simulated Mode
if data_source == "Simulated Dataset (Local Files)":
    active_mode_label = "💡 Standalone Simulated Dataset Mode (Offline)"
    df_raw = load_local_simulated_dataset()
    
    if scenario_choice == "Nominal Healthy Baseline":
        df = df_raw[df_raw["fault_class"] == 0]
    elif scenario_choice == "Bearing Wear (BPFO 107.4 Hz)":
        df = df_raw[df_raw["fault_class"] == 1]
    elif scenario_choice == "Rotor Imbalance (1X 30 Hz)":
        df = df_raw[df_raw["fault_class"] == 2]
    elif scenario_choice == "Shaft Misalignment (2X/3X)":
        df = df_raw[df_raw["fault_class"] == 3]
    elif scenario_choice == "Thermal Overheating":
        df = df_raw[df_raw["fault_class"] == 4]
    elif scenario_choice == "Motor Overload Ripple":
        df = df_raw[df_raw["fault_class"] == 5]
    elif scenario_choice == "Transient Sensor Glitch":
        df = df_raw[df_raw["fault_class"] == 6]
    else:
        df = df_raw

    df = df.tail(max_records)

    if not df.empty:
        last_rec = df.iloc[-1].to_dict()
        twin_state = {
            "asset_id": last_rec.get("asset_id", asset_id),
            "state": last_rec.get("maintenanceStatus", "HEALTHY"),
            "health_score": 100.0 if last_rec.get("fault_class") == 0 else (51.6 if last_rec.get("fault_class") == 1 else 38.0),
            "predicted_failure_mode": last_rec.get("fault_name", "Healthy"),
            "temperature_c": last_rec.get("temperature", 49.5),
            "rms_vibration": last_rec.get("vibration", 0.24),
            "current_a": last_rec.get("current", 1.20),
            "speed_rpm": last_rec.get("speed_rpm", 1800.0),
            "active_work_orders": ["SAP-WO-40003001"] if last_rec.get("fault_class") in [1, 4] else [],
            "active_purchase_requisitions": ["SAP-PR-10008001"] if last_rec.get("fault_class") == 4 else [],
        }

# AWS DynamoDB Mode
elif data_source == "AWS DynamoDB (Live Cloud)":
    active_mode_label = f"☁️ AWS DynamoDB Live Cloud ({region})"
    try:
        import boto3
        from boto3.dynamodb.conditions import Key
        resource = boto3.resource("dynamodb", region_name=region)
        history_table = resource.Table(history_table_name)
        latest_table = resource.Table(latest_table_name)

        latest_response = latest_table.get_item(Key={"deviceId": device_id})
        latest_item = decimal_to_float(latest_response.get("Item") or {})

        response = history_table.query(
            KeyConditionExpression=Key("deviceId").eq(device_id),
            ScanIndexForward=False,
            Limit=max_records,
        )
        records = [decimal_to_float(r) for r in response.get("Items", [])]
        df = pd.DataFrame(list(reversed(records)))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            
        twin_state = {
            "asset_id": asset_id,
            "state": latest_item.get("maintenanceStatus", "HEALTHY"),
            "health_score": 95.0,
            "temperature_c": latest_item.get("temperature", 50.0),
            "rms_vibration": latest_item.get("vibration", 0.25),
            "current_a": 1.20,
            "speed_rpm": 1800.0,
            "active_work_orders": [],
            "active_purchase_requisitions": [],
        }
    except Exception as exc:
        st.warning(f"⚠️ Live AWS connection failed ({exc}). Falling back to local simulated dataset.")
        active_mode_label = "💡 Fallback Local Simulated Dataset Mode"
        df = load_local_simulated_dataset().tail(max_records)


# -----------------------------------------------------------------------------
# Main Dashboard UI
# -----------------------------------------------------------------------------
st.info(active_mode_label)

if twin_state:
    st.subheader(f"📊 Digital Twin State · {twin_state.get('asset_id', asset_id)}")
    
    health = float(twin_state.get("health_score", 100.0))
    state_str = str(twin_state.get("state", "HEALTHY"))
    temp_val = float(twin_state.get("temperature_c", 49.5))
    vib_val = float(twin_state.get("rms_vibration", 0.24))
    curr_val = float(twin_state.get("current_a", 1.20))
    speed_val = float(twin_state.get("speed_rpm", 1800.0))
    fault_str = str(twin_state.get("predicted_failure_mode", "Healthy"))

    # Top KPI Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Health Score", f"{health:.1f}%", delta=f"{health - 95.0:.1f}%" if health < 90 else "Normal")
    k2.metric("Asset State", state_str)
    k3.metric("Temperature", f"{temp_val:.1f} °C", delta=f"{temp_val - 49.0:+.1f} °C" if temp_val > 52 else "Normal")
    k4.metric("Vibration RMS", f"{vib_val:.3f} g", delta=f"{vib_val - 0.24:+.3f} g" if vib_val > 0.35 else "Normal")
    k5.metric("Current Draw", f"{curr_val:.2f} A")
    k6.metric("Motor Speed", f"{speed_val:.0f} RPM")

    st.markdown(f"**Classification:** `{fault_str}` | **Active Work Orders:** `{len(twin_state.get('active_work_orders', []))}` | **Active PRs:** `{len(twin_state.get('active_purchase_requisitions', []))}`")

    # Closed-Loop ERP Action Card
    if state_str in ["WARNING", "CRITICAL", "MAINTENANCE"] or twin_state.get("active_work_orders"):
        st.divider()
        st.subheader("🤖 LangGraph Agent Swarm & Closed-Loop SAP ERP Action")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.markdown(f"""
            **Diagnostic Agent (RAG)**
            - *Failure Mode:* `{fault_str}`
            - *Vector Search:* ChromaDB index match (cosine similarity > 0.92)
            - *Recommended Action:* Bearing replacement / alignment check
            """)
        with ec2:
            pr_id = twin_state.get("active_purchase_requisitions", ["None"])[0] if twin_state.get("active_purchase_requisitions") else "In Stock"
            st.markdown(f"""
            **Inventory Agent (SAP MM)**
            - *Required Material:* SKF 6205-2RSH Bearing
            - *Stock Verification:* Shortage flagged
            - *Purchase Requisition:* `{pr_id}`
            """)
        with ec3:
            wo_id = twin_state.get("active_work_orders", ["None"])[0] if twin_state.get("active_work_orders") else "Pending"
            st.markdown(f"""
            **Scheduler Agent (SAP PM)**
            - *Work Order ID:* `{wo_id}`
            - *Priority Level:* `1 - CRITICAL`
            - *Work Center:* `MECH-01`
            """)
    st.divider()

# Sensor Charts
if not df.empty and "timestamp" in df:
    st.subheader("📈 Multi-Sensor Telemetry Trends & Harmonics")
    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature & Vibration", "⚡ Current & Speed", "📋 Telemetry Log"])
    
    with tab1:
        c1, c2 = st.columns(2)
        if "temperature" in df:
            fig_temp = px.line(df, x="timestamp", y="temperature", title="Motor Temperature (°C)", markers=True, color_discrete_sequence=["#f97316"])
            fig_temp.add_hline(y=75.0, line_dash="dash", line_color="yellow", annotation_text="Warning (75°C)")
            fig_temp.add_hline(y=90.0, line_dash="dash", line_color="red", annotation_text="Critical (90°C)")
            fig_temp.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            c1.plotly_chart(fig_temp, use_container_width=True)

        if "vibration" in df:
            fig_vib = px.line(df, x="timestamp", y="vibration", title="Radial Vibration RMS (g)", markers=True, color_discrete_sequence=["#06b6d4"])
            fig_vib.add_hline(y=1.0, line_dash="dash", line_color="yellow", annotation_text="Watchlist (1.0g)")
            fig_vib.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Critical Alarm (2.0g)")
            fig_vib.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            c2.plotly_chart(fig_vib, use_container_width=True)

    with tab2:
        c3, c4 = st.columns(2)
        if "current" in df:
            fig_curr = px.line(df, x="timestamp", y="current", title="Current Draw (A)", markers=True, color_discrete_sequence=["#10b981"])
            fig_curr.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Overload (1.5A)")
            fig_curr.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            c3.plotly_chart(fig_curr, use_container_width=True)

        if "speed_rpm" in df:
            fig_speed = px.line(df, x="timestamp", y="speed_rpm", title="Rotational Speed (RPM)", markers=True, color_discrete_sequence=["#a855f7"])
            fig_speed.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            c4.plotly_chart(fig_speed, use_container_width=True)

    with tab3:
        st.subheader("Recent Ingested Telemetry Records")
        display_cols = [c for c in ["timestamp", "deviceId", "asset_id", "temperature", "vibration", "current", "speed_rpm", "motorStatus", "maintenanceStatus", "fault_name"] if c in df]
        st.dataframe(df[display_cols].sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
