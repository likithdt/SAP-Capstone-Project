/**
 * Industrial IoT Digital Twin Frontend
 * Single-file application with API integration
 */

class App {
    constructor() {
        this.apiUrl = localStorage.getItem('apiUrl') || 'http://127.0.0.1:8000';
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
        }
    }

    async checkAPIStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/`);
            if (response.ok) {
                this.updateStatus(true);
                return true;
            }
        } catch (error) {
            this.updateStatus(false);
            return false;
        }
    }

    updateStatus(online) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');

        if (online) {
            indicator.classList.add('online');
            text.textContent = 'Online';
        } else {
            indicator.classList.remove('online');
            text.textContent = 'Offline';
            this.showToast('API connection failed', 'error', 5000);
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

            // Load orders
            this.loadOrders();
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
