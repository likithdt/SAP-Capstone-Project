# CI/CD Pipeline Architecture & Deployment Guide

This repository includes a production-grade **Continuous Integration & Continuous Deployment (CI/CD)** pipeline configured with **GitHub Actions**, **Docker containerization**, and a **Local Pre-Flight CI Runner**.

---

## 🏗️ Pipeline Architecture Overview

```
                                  +-------------------------------------------------+
                                  |                  Git Push / PR                  |
                                  +-------------------------------------------------+
                                                           |
                                                           v
+-------------------------------------------------------------------------------------------------------------------+
|                                              Continuous Integration (CI)                                          |
|                                                                                                                   |
|  [Stage 1: Lint & Syntax]  ──>  [Stage 2: Pytest Suite]  ──>  [Stage 3: Capstone Demo & KPI]  ──>  [Stage 4: Security] |
|  - Python compileall            - Simulator unit tests         - 4-scenario closed-loop demo       - Bandit AST scan  |
|  - Node.js JS validation        - Feature extraction tests     - KPI specification benchmark                          |
|  - Ruff static analysis         - State machine tests          (Accuracy > 85%, Latency < 5s)                         |
+-------------------------------------------------------------------------------------------------------------------+
                                                           |
                                                           | (On Merge to 'main' or Tag 'v*.*.*')
                                                           v
+-------------------------------------------------------------------------------------------------------------------+
|                                             Continuous Deployment (CD)                                            |
|                                                                                                                   |
|  [Job 1: Multi-Container Docker Build]          [Job 2: Serverless Packaging]         [Job 3: GitHub Release]     |
|  - ghcr.io/.../backend:latest                   - build/lambda-telemetry-handlers.zip - Automated release notes   |
|  - ghcr.io/.../streamlit:latest                                                       - Tagged release artifacts  |
+-------------------------------------------------------------------------------------------------------------------+
```

---

## 📁 CI/CD Configuration Files

| File | Purpose |
|------|---------|
| [`.github/workflows/ci.yml`](file:///.github/workflows/ci.yml) | GitHub Actions CI workflow: Linting, Unit testing, KPI benchmark gates, and Security scans |
| [`.github/workflows/cd.yml`](file:///.github/workflows/cd.yml) | GitHub Actions CD workflow: Multi-container Docker builds to GHCR, Lambda zip packaging, and Releases |
| [`Dockerfile`](file:///Dockerfile) | Production container for FastAPI Backend + Digital Twin Engine + HTML5 Web UI |
| [`Dockerfile.streamlit`](file:///Dockerfile.streamlit) | Production container for Streamlit Industrial IoT Analytics Dashboard |
| [`docker-compose.yml`](file:///docker-compose.yml) | Multi-container orchestration (FastAPI on `8000`, Streamlit on `8501`, optional Edge Simulator) |
| [`.dockerignore`](file:///.dockerignore) | Excludes virtualenvs, bytecode, test caches, and credential files |
| [`scripts/run_ci_checks.py`](file:///scripts/run_ci_checks.py) | Local pre-flight test runner that simulates the CI pipeline locally |

---

## ⚡ 1. How to Test CI Locally (Before Pushing)

You can run the full CI validation suite on your machine with one command:

```powershell
.\venv\Scripts\python.exe main.py ci
```

### Output:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Industrial IoT Digital Twin · Local CI/CD Pipeline Runner                   │
│ Simulating GitHub Actions CI Stages: Syntax, Tests, Demo & KPI Benchmarking │
└─────────────────────────────────────────────────────────────────────────────┘
▶ Running Stage: 1. Python Syntax & Compilation...
✔ PASSED: 1. Python Syntax & Compilation (0.13s)
▶ Running Stage: 2. Unit Tests (Simulator & Features)...
✔ PASSED: 2. Unit Tests (Simulator & Features) (6.14s)
▶ Running Stage: 3. Closed-Loop Jury Demonstration (4 Scenarios)...
✔ PASSED: 3. Closed-Loop Jury Demonstration (4 Scenarios) (6.71s)
▶ Running Stage: 4. Capstone KPI Benchmark Verification...
✔ PASSED: 4. Capstone KPI Benchmark Verification (7.07s)
                   CI/CD Pipeline Execution Summary                    
┌─────────────────────────────────────────────────┬──────────┬────────┐
│ Stage                                           │ Duration │ Status │
├─────────────────────────────────────────────────┼──────────┼────────┤
│ 1. Python Syntax & Compilation                  │ 0.13s    │ PASS   │
│ 2. Unit Tests (Simulator & Features)            │ 6.14s    │ PASS   │
│ 3. Closed-Loop Jury Demonstration (4 Scenarios) │ 6.71s    │ PASS   │
│ 4. Capstone KPI Benchmark Verification          │ 7.07s    │ PASS   │
└─────────────────────────────────────────────────┴──────────┴────────┘

🎉 All CI/CD pipeline checks passed successfully! Ready for push & deployment.
```

---

## 🐳 2. How to Run with Docker Compose

To build and run the entire multi-service prototype in isolated containers:

```bash
# Start FastAPI backend (port 8000) and Streamlit dashboard (port 8501)
docker compose up --build
```

### Accessing the Containerized Services:
* **Web Dashboard**: `http://localhost:8000/dashboard`
* **FastAPI OpenAPI Docs**: `http://localhost:8000/docs`
* **Streamlit Analytics Dashboard**: `http://localhost:8501`

### Running with Background Sensor Telemetry Simulation:
```bash
docker compose --profile simulation up --build
```
This automatically boots a 3rd container that continuously streams simulated physics telemetry into the backend.

---

## 🚀 3. GitHub Actions CI/CD Execution Details

### Continuous Integration (`ci.yml`)
Runs automatically on:
- Every `push` to `main`, `master`, or `develop`.
- Every `pull_request` against `main`, `master`, or `develop`.

**Jobs**:
1. **`lint-and-syntax`**: Compiles all Python modules and validates JavaScript syntax.
2. **`test-suite`**: Runs pytest across the test suite and verifies physics signal processing.
3. **`kpi-benchmark-and-demo`**: Executes the 4-scenario live jury demo and evaluates KPI targets.
4. **`security-audit`**: Runs AST-based vulnerability scanning using Bandit.

### Continuous Deployment (`cd.yml`)
Runs on push to `main` or release tag creation (`v1.0.0`):
1. **`build-and-push-docker`**: Builds multi-platform Docker images and pushes to GitHub Container Registry (`ghcr.io`).
2. **`package-lambda`**: Bundles `src/cloud/lambda_handlers.py` into a production-ready AWS Lambda deployment archive `lambda-telemetry-handlers.zip`.
3. **`create-release`**: When a version tag (e.g., `git tag v1.0.0 && git push --tags`) is pushed, publishes a GitHub Release with the deployment artifacts attached.
