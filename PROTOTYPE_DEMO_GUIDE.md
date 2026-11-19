# Prototype Demonstration Guide (Simulated Dataset Mode)

This guide provides everything you need to showcase the **Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine with Closed-Loop ERP Action** using the simulated dataset—without requiring live physical sensors or active AWS credentials.

---

## ⚡ The Two Dashboards Work Together in Real-Time

Both the **Streamlit Analytics App** and the **Web Frontend** are now fully integrated and stay synchronized through the FastAPI Digital Twin engine:

1. **FastAPI Backend (`port 8000`)** maintains the single source of truth:
   - Digital Twin state machine (`HEALTHY`, `WARNING`, `CRITICAL`, `MAINTENANCE`)
   - Composite Health Score engine ($0-100\%$)
   - Rolling multi-sensor time-series history
   - SAP PM Work Orders & SAP MM Purchase Requisitions ledgers
2. **Streamlit App (`port 8501`)** now defaults to `FastAPI Digital Twin (Live Sync with Frontend)`:
   - Reflects state and faults triggered from the frontend in real time.
   - Any fault injected in Streamlit immediately updates the FastAPI backend and Web Frontend.
   - Also directly embedded inside the Web Frontend under the **"Streamlit App"** tab!
3. **Web Frontend (`http://127.0.0.1:8000/dashboard` or `port 3000`)**:
   - Modern dark UI with live rolling Canvas charts, one-click fault injection presets, and automated SAP dispatching.

---

## 🚀 Step-by-Step: How to Run the Prototype

### Step 1: Start the FastAPI Backend (Terminal 1)
```powershell
.\venv\Scripts\python.exe main.py serve --port 8000
```
*The Web Dashboard is immediately available at: **`http://127.0.0.1:8000/dashboard`***.

### Step 2: Start the Streamlit Analytics App (Terminal 2)
```powershell
.\venv\Scripts\python.exe main.py dashboard --port 8501
```
*Streamlit is available at **`http://localhost:8501`** and also accessible right inside the Web Dashboard's **"Streamlit App"** tab!*

---

## 🎯 What to Showcase During Your Demo

### In the Web Frontend (`http://127.0.0.1:8000/dashboard`)
* **Overview Tab**:
  - Live rolling multi-sensor Canvas charts for **Temperature** and **Vibration RMS**.
  - Click **"Start Live Simulation Stream"** to show hands-free dynamic telemetry streaming every 3 seconds.
  - Click any preset button (e.g. `1: Bearing Defect (BPFO)`) to demonstrate:
    1. Anomaly detection & diagnosis by LangGraph Diagnostic Agent.
    2. Health score dropping from $100\%$ to $\approx 51.6\%$.
    3. State transition to `MAINTENANCE`.
    4. Automated SAP PM Work Order generation (`#40003001`).
  - Click **"Reset"** to return the digital twin to $100\%$ baseline.
* **Streamlit App Tab**:
  - Click the **"Streamlit App"** tab in the top navigation to view the live Streamlit dashboard embedded directly inside the frontend!

### In Streamlit (`http://localhost:8501`)
* In the sidebar, verify the data source mode is **`FastAPI Digital Twin (Live Sync with Frontend)`**.
* Notice the green banner: `🟢 Connected to FastAPI Server (http://127.0.0.1:8000) · Synced with Web Frontend`.
* Click **"🚨 Bearing Defect"** or **"🔥 Overheating"** to inject a fault:
  - Both Streamlit and the Web Frontend immediately update with the new telemetry and created SAP work orders!
* Explore the **FFT Dominant Frequency Harmonics** and multi-sensor Plotly charts with safety threshold lines.
     - ⚡ `6: Sensor Glitch` -> Discriminates sensor spike without false mechanical work order.
   - **Live Simulation Stream**: Click **"Start Live Simulation Stream"** to watch continuous live telemetry update the graphs and health score every 3 seconds hands-free!
   - **Reset Button**: Click **"Reset"** to return the digital twin and SAP ledgers to 100% nominal baseline.

---

### Option 3: Automated CLI Live Jury Demonstration

Executes all 4 Capstone operational jury scenarios in sequence and verifies 100% closed-loop fidelity:

```powershell
.\venv\Scripts\python.exe main.py demo
```

**Output**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SAP Capstone Project Live Jury Demonstration                                │
│ Closed-Loop ERP Action: IoT Anomaly -> AI Diagnosis -> LangGraph Swarm ->   │
│ SAP PM/MM Action                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
[PASS] Scenario 1: Asset MOTOR-001 operating at Health Score 92.1 (HEALTHY). Zero false work orders.
[PASS] Scenario 2: Detected Bearing Wear (BPFO) (Health: 51.6). Created SAP PM Work Order #40003001. State: MAINTENANCE.
[PASS] Scenario 3: Severe Thermal Drift. Spare part out of stock! Generated SAP MM Purchase Requisition #10008001 & Priority 1 Work Order #40003002.
[PASS] Scenario 4: Correctly isolated Sensor Anomaly. Mechanical bearing replacement successfully avoided.

*** All 4 Capstone jury scenarios executed with 100% closed-loop fidelity! ***
```

---

## 📡 Standalone Sensor Simulator (CLI)

You can run the simulated edge device in a separate terminal to stream data into the local dataset or FastAPI backend:

```powershell
# Continuous local telemetry stream (appends to data/simulated_telemetry_history.csv)
.\venv\Scripts\python.exe amazon/simulator/sensor_simulator.py --mode local --fault healthy --interval 2.0

# Inject specific bearing fault
.\venv\Scripts\python.exe amazon/simulator/sensor_simulator.py --mode local --fault bearing --once

# Stream simulated telemetry directly into FastAPI server
.\venv\Scripts\python.exe amazon/simulator/sensor_simulator.py --mode api --fault bearing --interval 2.0
```

---

## 📁 Generated Simulated Dataset Files

The following dataset files are generated and ready in the repository:

| File | Description | Records |
|------|-------------|---------|
| `data/simulated_telemetry_history.csv` | Chronological multi-sensor time series across all 7 fault classes | 340+ rows |
| `data/simulated_telemetry_history.json` | JSON format of historical telemetry | 340+ records |
| `data/simulated_telemetry_batch.json` | High-frequency 1000 Hz waveform bursts (`vibration_x,y,z`) | 340 batches |
| `data/scenarios/scenario_1_healthy.json` | Preset nominal baseline telemetry | 10 samples |
| `data/scenarios/scenario_2_bearing_defect.json` | Preset outer race BPFO bearing defect telemetry | 10 samples |
| `data/scenarios/scenario_3_thermal_stockout.json` | Preset thermal runaway + spare part stockout | 10 samples |
| `data/scenarios/scenario_4_sensor_glitch.json` | Preset transient sensor spike telemetry | 10 samples |

To re-generate or expand the dataset with custom sample sizes:
```powershell
.\venv\Scripts\python.exe scripts/generate_simulated_dataset.py
# or
.\venv\Scripts\python.exe main.py generate-dataset --samples 100
```

---

## 📊 Summary of Simulated Operational Fault Classes

| Class # | Operational Fault Profile | Physical Characteristics | Closed-Loop ERP Action |
|:-------:|:-------------------------|:--------------------------|:-----------------------|
| **0** | **Nominal Healthy** | 1800 RPM, 48–51°C, 1.2A, RMS vibration < 0.3g | Zero false alarms, State: `HEALTHY` |
| **1** | **Bearing Wear (BPFO)** | BPFO impact frequency at 107.4 Hz, RMS > 2.0g | Auto SAP PM Work Order (Priority 2) |
| **2** | **Rotor Imbalance** | Sinusoidal dominant 1X peak at 30 Hz | State: `WATCHLIST`, vibration advisory |
| **3** | **Shaft Misalignment** | 2X (60 Hz) and 3X (90 Hz) harmonic peaks | State: `WARNING`, alignment work order |
| **4** | **Thermal Overheating** | Temperature > 95°C, thermal drift, spare part stockout | SAP PM Order (Priority 1) + SAP MM Purchase Requisition |
| **5** | **Motor Overload** | Current > 1.8A, 100 Hz supply ripple | Motor load alert, State: `CRITICAL` |
| **6** | **Sensor Glitch** | High single-sample spike, nominal temp/current | Sensor calibration ticket, avoids bearing teardown |
