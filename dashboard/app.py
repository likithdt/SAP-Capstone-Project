import os
from datetime import datetime, timezone, timedelta

import boto3
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from boto3.dynamodb.conditions import Key

load_dotenv()

st.set_page_config(page_title="Industrial IoT Digital Twin", page_icon="🏭", layout="wide")
st.title("🏭 Industrial IoT Digital Twin")
st.caption("Software-only telemetry demo · Data is simulated")

region = os.getenv("AWS_REGION", "ap-south-1")
history_table_name = os.getenv("DYNAMODB_TABLE", "MachineTelemetry")
latest_table_name = os.getenv("LATEST_TABLE_NAME", "MachineLatestState")

@st.cache_resource
def get_tables():
    resource = boto3.resource("dynamodb", region_name=region)
    return resource.Table(history_table_name), resource.Table(latest_table_name)

def decimal_to_float(value):
    if isinstance(value, dict):
        return {k: decimal_to_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decimal_to_float(v) for v in value]
    try:
        # Decimal can be converted to float; strings and ints are handled safely.
        from decimal import Decimal
        if isinstance(value, Decimal):
            return float(value)
    except Exception:
        pass
    return value

with st.sidebar:
    st.header("Controls")
    device_id = st.text_input("Device ID", "industrial-machine-01")
    max_records = st.slider("Maximum history records", 20, 500, 100, step=20)
    refresh = st.button("🔄 Refresh data", type="primary")
    st.caption(f"AWS region: {region}")
    st.caption(f"History table: {history_table_name}")

try:
    history_table, latest_table = get_tables()
    latest_response = latest_table.get_item(Key={"deviceId": device_id})
    latest = latest_response.get("Item")

    # Query the device partition, newest records first.
    response = history_table.query(
        KeyConditionExpression=Key("deviceId").eq(device_id),
        ScanIndexForward=False,
        Limit=max_records,
    )
    records = response.get("Items", [])
    # Query returns newest first; chart displays chronological order.
    records = list(reversed(records))
    records = [decimal_to_float(r) for r in records]
    df = pd.DataFrame(records)

    if latest:
        latest = decimal_to_float(latest)
        st.subheader("Latest machine state")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f"{latest.get('temperature', '—')} °C")
        c2.metric("Vibration", f"{latest.get('vibration', '—')}")
        c3.metric("Humidity", f"{latest.get('humidity', '—')} %")
        c4.metric("Maintenance", latest.get("maintenanceStatus", "UNKNOWN"))
        st.write(f"**Motor status:** {latest.get('motorStatus', 'UNKNOWN')}")
        st.write(f"**Last timestamp:** {latest.get('timestamp', 'Unknown')}")
    else:
        st.info("No latest-state record found yet. Start the simulator and refresh.")

    if df.empty:
        st.warning("No telemetry history found for this device. Start the simulator, wait for a few messages, then click Refresh.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        st.subheader("Sensor history")
        chart_cols = st.columns(2)
        if "temperature" in df:
            chart_cols[0].plotly_chart(
                px.line(df, x="timestamp", y="temperature", title="Temperature (°C)", markers=True),
                use_container_width=True,
            )
        if "vibration" in df:
            chart_cols[1].plotly_chart(
                px.line(df, x="timestamp", y="vibration", title="Vibration", markers=True),
                use_container_width=True,
            )
        if "humidity" in df:
            st.plotly_chart(
                px.line(df, x="timestamp", y="humidity", title="Humidity (%)", markers=True),
                use_container_width=True,
            )
        st.subheader("Recent telemetry records")
        display_cols = [c for c in ["timestamp", "deviceId", "temperature", "vibration", "humidity", "motorStatus", "maintenanceStatus", "ingestedAt"] if c in df]
        st.dataframe(df[display_cols].sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

        critical = df[df.get("maintenanceStatus", pd.Series(index=df.index, dtype=str)).isin(["WARNING", "CRITICAL"])]
        st.subheader("Maintenance events")
        if critical.empty:
            st.success("No warning or critical records in the displayed window.")
        else:
            st.dataframe(critical[display_cols].sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

except Exception as exc:
    st.error("Could not read AWS data. Check AWS CLI credentials, region, table names, and DynamoDB permissions.")
    st.code(str(exc))
    st.markdown("Run `aws sts get-caller-identity` in PowerShell to verify your AWS CLI identity.")
