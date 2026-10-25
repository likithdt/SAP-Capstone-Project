# Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine with Closed-Loop ERP Action

## Implementation Status

- [x] **Step 1: Physical Asset Simulation & Synthetic Data Generation** (`src/digital_twin/simulator.py`, 7 operational classes, Nyquist sampling, physics-based fault profiles)
- [x] **Step 2: Signal Processing & Feature Engineering** (`src/features/`, time-domain statistics, FFT spectral analysis, envelope demodulation, thermal/current metrics)
- [x] **Step 3: Deep Learning Multimodal Prognostics Engine** (`src/models/`, Isolation Forest baseline, LSTM Autoencoder, 1D-CNN Fault Classifier, CNN-LSTM RUL Regressor)
- [x] **Step 4: MLOps Tracking & Drift Monitoring** (`src/monitoring/`, MLflow experiment tracking & model registry, Evidently AI data drift, prediction drift, sensor data quality checks, closed-loop automated retraining workflows)
- [x] **Step 5: Agentic AI Decision Swarm** (`src/agents/`, LangGraph multi-agent orchestration: Diagnostic Agent with ChromaDB vector-search RAG, Inventory Agent with SAP MM stock verification & automated Purchase Requisition generation, Scheduler Agent with failure criticality prioritization & SAP PM payload generation)
- [x] **Step 6: SAP Enterprise Core & Digital Twin Fiori UI** (`src/sap/client.py`, `src/digital_twin/state_machine.py`, `src/digital_twin/fiori_adapter.py`, Section 9.3 HealthScore engine, Section 9.2 state machine transitions, SAP BTP OAuth 2.0 & S/4HANA OData v4 PM/MM client dispatch, SAP Fiori Overview Page analytical & operational cards)
- [x] **Step 7: Edge Sensing, MQTT Telemetry & Cloud Ingestion** (`src/edge/`, `src/cloud/`, ESP32 C++ firmware with local SD failover, Wokwi simulation circuit `diagram.json`, Python Edge MQTT client, S3 / Timestream / DynamoDB cloud storage adapters, and all 5 serverless AWS Lambda handlers)
- [x] **Step 8: Fault Injection Benchmarking, KPI Evaluation & Jury Demo Presentation** (`src/benchmarks/`, `main.py`, `src/api.py`, systematic fault injection across all 7 classes, confusion matrix, RUL RMSE regression, end-to-end latency benchmarks < 5s, SAP work order success rate > 95%, and automated jury demo runner)

## Test Coverage
- **Total Tests Passing**: 94/94 across unit, functional, signal processing, deep learning, MLOps, LangGraph multi-agent swarm, SAP Enterprise OData, Digital Twin state machine, Serverless AWS Lambda, KPI Benchmark, and FastAPI REST suites (`pytest tests/`).

---
### CLI & API Usage

```bash
# Run automated live demonstration suite across all 4 jury scenarios
python main.py demo

# Run comprehensive KPI evaluation benchmarks
python main.py benchmark --samples 3

# Start FastAPI Digital Twin and SAP Fiori REST server
python main.py serve --port 8000

# Run full test suite
python main.py test
```
