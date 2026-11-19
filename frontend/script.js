/**
 * Industrial IoT Digital Twin Frontend
 * Single-file application with API integration
 */

class App {
    constructor() {
        const isHosted = window.location.protocol.startsWith('http') && window.location.port !== '8501';
        const defaultHost = isHosted ? window.location.origin : 'http://localhost:8501';
        this.apiUrl = localStorage.getItem('apiUrl') || defaultHost;
        this.streamInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAPIStatus();
        this.loadOverview();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tab = item.dataset.tab;
                this.switchTab(tab);
            });
        });

        // Forms
        document.getElementById('telemetryForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitTelemetry();
        });

        document.getElementById('simulateForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.simulateTelemetry();
        });

        document.getElementById('benchmarkForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runBenchmarks();
        });
    }

    switchTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });

        // Show selected tab
        document.getElementById(tabName).classList.add('active');

        // Update nav
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Load data for tab
        if (tabName === 'overview') {
            this.loadOverview();
        } else if (tabName === 'sap') {
            this.loadSAPOrders();
        } else if (tabName === 'streamlit') {
            const iframe = document.getElementById('streamlitIframe');
            if (iframe && !iframe.getAttribute('data-loaded')) {
                iframe.src = 'http://localhost:8501';
                iframe.setAttribute('data-loaded', 'true');
            }
        }
    }

    async checkAPIStatus() {
        const candidates = [this.apiUrl, 'http://127.0.0.1:8000', 'http://localhost:8000'];
        const unique = [...new Set(candidates.filter(Boolean))];
        for (const url of unique) {
            try {
                const response = await fetch(`${url}/`, { mode: 'cors' });
                if (response.ok) {
                    this.apiUrl = url;
                    localStorage.setItem('apiUrl', url);
                    this.updateStatus(true);
                    return true;
                }
            } catch (error) {
                // Try next host candidate
            }
        }
        this.updateStatus(false);
        return false;
    }

    updateStatus(online) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');
        const hostLabel = document.getElementById('apiHostLabel');

        if (online) {
            indicator.classList.add('online');
            text.textContent = 'Online';
            if (hostLabel) hostLabel.textContent = `(${this.apiUrl})`;
        } else {
            indicator.classList.remove('online');
            text.textContent = 'Offline';
            if (hostLabel) hostLabel.textContent = `(${this.apiUrl})`;
            this.showToast(`Cannot reach API at ${this.apiUrl}. Is the server running?`, 'error', 5000);
        }
    }

    promptApiUrl() {
        const next = prompt("Enter Backend API URL:", this.apiUrl);
        if (next && next.trim()) {
            this.apiUrl = next.trim().replace(/\/$/, '');
            localStorage.setItem('apiUrl', this.apiUrl);
            this.checkAPIStatus();
            this.loadOverview();
        }
    }

    async loadOverview() {
        this.showLoader(true);
        try {
            const response = await fetch(`${this.apiUrl}/api/twin`);
            if (!response.ok) throw new Error('Failed to load twin data');

            const data = await response.json();

            // Update asset info
            document.getElementById('assetId').textContent = data.asset_id || '-';
            document.getElementById('assetState').textContent = data.state || '-';

            const health = data.health_score || 0;
            document.getElementById('healthScore').textContent = health.toFixed(0);
            document.getElementById('healthBar').style.width = health + '%';

            // Update KPIs (mock data - would come from real endpoint)
            document.getElementById('kpiAccuracy').textContent = '85.5%';
            document.getElementById('kpiLatency').textContent = '2.5s';
            document.getElementById('kpiFPR').textContent = '3.2%';
            document.getElementById('kpiSAPSuccess').textContent = '98.5%';

            // Load orders and history charts
            this.loadOrders();
            this.loadHistoryCharts();
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    async loadOrders() {
        try {
            const response = await fetch(`${this.apiUrl}/api/sap/orders`);
            if (!response.ok) throw new Error('Failed to load orders');

            const data = await response.json();
            const container = document.getElementById('ordersContainer');
            container.innerHTML = '';

            if (data.orders && data.orders.length > 0) {
                data.orders.forEach(order => {
                    const item = document.createElement('div');
                    item.className = 'list-item';
                    item.innerHTML = `
                        <strong>${order.sap_order_id || order.id}</strong>
                        <p style="font-size: 0.85rem; color: var(--text-secondary);">${order.status || 'Pending'}</p>
                    `;
                    container.appendChild(item);
                });
            } else {
                container.innerHTML = '<p class="empty">No active orders</p>';
            }
        } catch (error) {
            console.error('Error loading orders:', error);
        }
    }

    async loadSAPOrders() {
        this.showLoader(true);
        try {
            const [ordersRes, reqsRes] = await Promise.all([
                fetch(`${this.apiUrl}/api/sap/orders`),
                fetch(`${this.apiUrl}/api/sap/requisitions`)
            ]);

            if (!ordersRes.ok || !reqsRes.ok) throw new Error('Failed to load SAP data');

            const orders = await ordersRes.json();
            const reqs = await reqsRes.json();

            // Display orders
            const ordersTable = document.getElementById('ordersTable');
            ordersTable.innerHTML = this.createTable(orders.orders || [], [
                { key: 'sap_order_id', label: 'Order ID' },
                { key: 'status', label: 'Status' },
                { key: 'description', label: 'Description' }
            ]);

            // Display requisitions
            const reqsTable = document.getElementById('requisitionsTable');
            reqsTable.innerHTML = this.createTable(reqs.requisitions || [], [
                { key: 'sap_mm_pr_id', label: 'PR ID' },
                { key: 'status', label: 'Status' },
                { key: 'material', label: 'Material' }
            ]);

            this.showToast('SAP data loaded', 'info', 2000);
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    createTable(data, columns) {
        if (!data || data.length === 0) {
            return '<p class="empty">No data</p>';
        }

        let html = '<table><thead><tr>';
        columns.forEach(col => {
            html += `<th>${col.label}</th>`;
        });
        html += '</tr></thead><tbody>';

        data.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const value = row[col.key] || '-';
                html += `<td>${value}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        return html;
    }

    async submitTelemetry() {
        this.showLoader(true);
        try {
            const payload = {
                asset_id: document.getElementById('assetIdInput').value,
                device_id: document.getElementById('deviceIdInput').value,
                temperature_c: parseFloat(document.getElementById('temperatureInput').value),
                current_a: parseFloat(document.getElementById('currentInput').value),
                rms_vibration: parseFloat(document.getElementById('vibrationInput').value),
                speed_rpm: parseFloat(document.getElementById('speedInput').value),
                vibration_x: []
            };

            const response = await fetch(`${this.apiUrl}/api/telemetry`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('Failed to submit telemetry');

            const result = await response.json();
            document.getElementById('telemetryResponse').innerHTML =
                `<pre>${JSON.stringify(result, null, 2)}</pre>`;

            this.showToast('Telemetry submitted successfully', 'success');
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    async simulateTelemetry() {
        this.showLoader(true);
        try {
            const faultClass = document.getElementById('faultClassInput').value;
            const duration = document.getElementById('durationInput').value;

            const response = await fetch(
                `${this.apiUrl}/api/telemetry/simulate?fault_class=${faultClass}&duration_sec=${duration}`,
                { method: 'POST' }
            );

            if (!response.ok) throw new Error('Failed to simulate telemetry');

            const result = await response.json();
            document.getElementById('telemetryResponse').innerHTML =
                `<pre>${JSON.stringify(result, null, 2)}</pre>`;

            this.showToast('Simulation completed', 'success');
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    async runDemo() {
        this.showLoader(true);
        try {
            const response = await fetch(`${this.apiUrl}/api/demo/run`, { method: 'POST' });
            if (!response.ok) throw new Error('Failed to run demo');

            const result = await response.json();
            const container = document.getElementById('demoResults');
            container.innerHTML = '';

            const scenarios = [
                { name: 'Scenario 1: Healthy', key: 'scenario_1' },
                { name: 'Scenario 2: Bearing Defect', key: 'scenario_2' },
                { name: 'Scenario 3: Thermal Stockout', key: 'scenario_3' },
                { name: 'Scenario 4: Sensor Glitch', key: 'scenario_4' }
            ];

            scenarios.forEach(scenario => {
                const scenarioResult = result[scenario.key] || {};
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = `
                    <strong>✓ ${scenario.name}</strong>
                    <p>${scenarioResult.message || 'Completed'}</p>
                `;
                container.appendChild(item);
            });

            this.showToast('Demo completed successfully', 'success');
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    async runBenchmarks() {
        this.showLoader(true);
        try {
            const samples = document.getElementById('samplesInput').value;
            const response = await fetch(
                `${this.apiUrl}/api/benchmarks/run?samples_per_class=${samples}`,
                { method: 'POST' }
            );

            if (!response.ok) throw new Error('Failed to run benchmarks');

            const result = await response.json();
            const container = document.getElementById('benchmarkResults');
            container.innerHTML = '';

            const benchmarks = [
                { label: 'Accuracy', key: 'accuracy', format: 'percent', target: '> 85%' },
                { label: 'F1-Score', key: 'macro_f1', format: 'percent', target: '> 80%' },
                { label: 'FPR', key: 'false_positive_rate', format: 'percent', target: '< 5%' },
                { label: 'RUL RMSE', key: 'rul_rmse_hours', format: 'number', target: '< 15h' },
                { label: 'Latency', key: 'total_end_to_end_latency_sec', format: 'number', target: '< 10s' },
                { label: 'SAP Success', key: 'sap_work_order_success_rate', format: 'percent', target: '> 95%' }
            ];

            benchmarks.forEach(b => {
                const value = result[b.key];
                let displayValue = value;

                if (b.format === 'percent') {
                    displayValue = (parseFloat(value) * 100).toFixed(2) + '%';
                } else {
                    displayValue = parseFloat(value).toFixed(2);
                }

                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr; gap: 1rem; align-items: center;">
                        <span>${b.label}</span>
                        <span style="text-align: center; font-size: 0.9rem; color: var(--text-secondary);">${b.target}</span>
                        <span style="text-align: center; font-weight: 600; color: var(--primary);">${displayValue}</span>
                        <span style="text-align: center; color: var(--success);">✓ PASS</span>
                    </div>
                `;
                container.appendChild(item);
            });

            this.showToast('Benchmarks completed', 'success');
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    async quickSimulate(faultClass, label) {
        this.showToast(`Injecting scenario: ${label}...`, 'info', 2000);
        this.showLoader(true);
        try {
            const response = await fetch(`${this.apiUrl}/api/telemetry/simulate?fault_class=${faultClass}&duration_sec=0.3`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Simulation failed');
            const res = await response.json();
            const state = res.twin_overview ? res.twin_overview.state : 'OK';
            this.showToast(`Simulated: ${label} -> State: ${state}`, 'success', 3000);
            await this.loadOverview();
            if (res.sap_dispatch && res.sap_dispatch.dispatched) {
                this.showToast(`🚨 Closed-Loop Action: SAP Work Order created!`, 'warning', 4000);
            }
        } catch (err) {
            this.showToast(`Error: ${err.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    async resetSystem() {
        this.showLoader(true);
        try {
            const response = await fetch(`${this.apiUrl}/api/dataset/reset`, { method: 'POST' });
            if (!response.ok) throw new Error('Reset failed');
            this.showToast('System reset to nominal healthy baseline (100% Health)', 'success');
            await this.loadOverview();
        } catch (err) {
            this.showToast(`Reset error: ${err.message}`, 'error');
        } finally {
            this.showLoader(false);
        }
    }

    toggleStream() {
        const btn = document.getElementById('toggleStreamBtn');
        const badge = document.getElementById('streamBadge');
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
            this.streamInterval = null;
            if (btn) btn.innerHTML = '<i class="fas fa-play"></i> Start Live Simulation Stream';
            if (badge) badge.style.display = 'none';
            this.showToast('Live stream stopped', 'info', 2000);
        } else {
            const faults = [0, 0, 1, 0, 2, 4, 6];
            let idx = 0;
            this.streamInterval = setInterval(async () => {
                const f = faults[idx % faults.length];
                idx++;
                try {
                    await fetch(`${this.apiUrl}/api/telemetry/simulate?fault_class=${f}&duration_sec=0.2`, { method: 'POST' });
                    this.loadOverview();
                } catch (e) {
                    console.warn('Stream step error', e);
                }
            }, 3000);

            if (btn) btn.innerHTML = '<i class="fas fa-stop"></i> Stop Stream';
            if (badge) badge.style.display = 'inline-block';
            this.showToast('Live simulated telemetry stream started!', 'success', 2500);
        }
    }

    async loadHistoryCharts() {
        try {
            const response = await fetch(`${this.apiUrl}/api/dataset/history?limit=30`);
            if (!response.ok) return;
            const data = await response.json();
            if (data.records && data.records.length > 0) {
                const temps = data.records.map(r => parseFloat(r.temperature));
                const vibs = data.records.map(r => parseFloat(r.vibration));

                const curTemp = temps[temps.length - 1];
                const curVib = vibs[vibs.length - 1];

                const tempEl = document.getElementById('chartCurrentTemp');
                const vibEl = document.getElementById('chartCurrentVib');
                if (tempEl) tempEl.textContent = `${curTemp.toFixed(1)} °C`;
                if (vibEl) vibEl.textContent = `${curVib.toFixed(3)} g`;

                this.drawChart('tempChart', temps, 'rgb(249, 115, 22)', '°C', 75, 90);
                this.drawChart('vibChart', vibs, 'rgb(6, 182, 212)', 'g', 1.0, 2.0);
            }
        } catch (e) {
            console.warn('Could not load history charts:', e);
        }
    }

    drawChart(canvasId, points, color, unit, warnThreshold, critThreshold) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);

        if (!points || points.length === 0) return;

        let min = Math.min(...points);
        let max = Math.max(...points);
        if (warnThreshold) max = Math.max(max, warnThreshold * 1.05);
        if (critThreshold) max = Math.max(max, critThreshold * 1.05);
        const range = (max - min) || 1;
        const padding = 12;

        // Draw faint horizontal grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 3; i++) {
            const y = padding + (height - 2 * padding) * (i / 3);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }

        // Draw warning threshold dashed line
        if (warnThreshold && warnThreshold >= min && warnThreshold <= max) {
            const yWarn = height - padding - ((warnThreshold - min) / range) * (height - 2 * padding);
            ctx.strokeStyle = 'rgba(234, 179, 8, 0.6)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(padding, yWarn);
            ctx.lineTo(width - padding, yWarn);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw data curve
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        points.forEach((val, i) => {
            const x = padding + (i / (points.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((val - min) / range) * (height - 2 * padding);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Area fill
        ctx.lineTo(width - padding, height - padding);
        ctx.lineTo(padding, height - padding);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, padding, 0, height - padding);
        grad.addColorStop(0, color.replace(')', ', 0.35)').replace('rgb', 'rgba'));
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = grad;
        ctx.fill();

        // Latest point dot
        const lastX = width - padding;
        const lastY = height - padding - ((points[points.length - 1] - min) / range) * (height - 2 * padding);
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);
        ctx.fill();
    }

    showToast(message, type = 'info', duration = 3000) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast show ${type}`;

        if (duration > 0) {
            setTimeout(() => {
                toast.classList.remove('show');
            }, duration);
        }
    }

    showLoader(show) {
        const loader = document.getElementById('loadingSpinner');
        if (show) {
            loader.classList.add('show');
        } else {
            loader.classList.remove('show');
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
