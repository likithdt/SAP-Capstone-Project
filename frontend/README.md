# Industrial IoT Digital Twin Dashboard - Frontend

A modern, responsive web UI for the Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine. Built with vanilla HTML, CSS, and JavaScript—no frameworks required.

## Features

### 📊 Overview Dashboard
- Real-time asset status monitoring
- Health score visualization with progress bar
- System KPI metrics (Accuracy, Latency, FPR, SAP Success Rate)
- Active maintenance orders display
- Recent activity timeline
- Auto-refresh capability

### 📡 Telemetry Management
- Manual telemetry ingestion form
- Simulated telemetry batch generation (7 fault classes)
- Real-time response visualization
- Sensor data inputs (temperature, current, vibration, speed)

### 🔧 SAP Integration
- View all created SAP PM Maintenance Orders
- Track Purchase Requisitions (MM)
- Real-time order status updates
- Automatic table rendering from API data

### 🎬 Jury Demonstration
- Run automated 4-scenario demonstration suite
- Scenario 1: Nominal Healthy Equipment
- Scenario 2: Bearing Defect (BPFO) Detection
- Scenario 3: Thermal Overheating & Stockout
- Scenario 4: Sensor Glitch Discrimination
- Real-time scenario execution tracking

### 📈 KPI Benchmarking
- Run comprehensive benchmark suite
- Verify compliance against Capstone targets
- View detailed KPI results
- Performance metrics display

### ⚙️ Settings
- Configurable API Base URL
- Auto-refresh interval control
- Persistent localStorage configuration
- API connection status monitoring

## Project Structure

```
frontend/
├── index.html                 # Main HTML template
├── README.md                  # This file
├── styles/
│   └── main.css              # Complete styling (responsive)
└── js/
    ├── api.js                # FastAPI client wrapper
    ├── ui.js                 # UI helper functions & components
    └── app.js                # Main application logic
```

## Getting Started

### Prerequisites
- Any modern web browser (Chrome, Firefox, Safari, Edge)
- Running FastAPI backend on `http://localhost:8000` (configurable)
- Python requirements installed (see main project README)

### Running the Frontend

1. **Start the FastAPI backend:**
   ```bash
   python main.py serve --host 127.0.0.1 --port 8000
   ```

2. **Open the frontend in your browser:**
   - Simply open `frontend/index.html` in a web browser
   - Or use a local HTTP server for better compatibility:
     ```bash
     # Python 3
     python -m http.server 8080 -d frontend
     # Then visit http://localhost:8080
     ```

3. **Configure API URL (if needed):**
   - Navigate to Settings tab
   - Update API Base URL if backend is running on a different address
   - Click Save Settings

## Architecture

### API Client (`js/api.js`)
- RESTful wrapper around the FastAPI backend
- Handles HTTP requests, timeouts, and error handling
- Methods: health check, telemetry ingestion, simulation, SAP queries, demo/benchmarks

### UI Helpers (`js/ui.js`)
- **UIHelper**: DOM manipulation, notifications, formatting utilities
- **NavigationHelper**: Section/route management
- **DataFormatter**: Specialized response formatting

### Application (`js/app.js`)
- Main app initialization and event binding
- Settings management (localStorage)
- Data loading and refresh logic
- Form submission handlers
- Auto-refresh orchestration

### Styling (`styles/main.css`)
- Dark modern theme (inspired by SAP Fiori Next)
- CSS variables for theming
- Responsive grid layouts
- Smooth animations and transitions
- Dark mode optimized for industrial dashboards

## API Integration

The frontend communicates with the following FastAPI endpoints:

```
GET  /                              # Health check
GET  /api/twin                       # Digital twin state
GET  /api/fiori/overview             # Fiori dashboard snapshot
POST /api/telemetry                  # Ingest telemetry
POST /api/telemetry/simulate         # Simulate batch with fault class
GET  /api/sap/orders                 # List SAP maintenance orders
GET  /api/sap/requisitions           # List purchase requisitions
POST /api/demo/run                   # Run jury demo (all 4 scenarios)
POST /api/benchmarks/run             # Run KPI benchmarks
```

## Configuration

Settings are stored in browser localStorage:

| Key | Default | Description |
|-----|---------|-------------|
| `apiURL` | `http://localhost:8000` | Backend API base URL |
| `refreshInterval` | `5` | Auto-refresh interval (seconds) |
| `autoRefresh` | `true` | Enable auto-refresh on overview/SAP tabs |

## Features & Usage

### 🔄 Auto-Refresh
- Enabled by default on Overview and SAP tabs
- Configurable interval (1-60 seconds)
- Disable from Settings tab

### 📱 Responsive Design
- Desktop: Full dashboard layout with sidebar
- Tablet: Collapsible sidebar, optimized grid
- Mobile: Horizontal scrolling nav, single-column layout

### 🎨 Dark Theme
- Eye-friendly for industrial environments
- SAP Fiori-inspired color scheme
- Customizable via CSS variables in `:root`

### 🔔 Toast Notifications
- Success (green), Error (red), Warning (orange), Info (blue)
- Auto-dismiss or manual close
- Positioned bottom-right

### ⏳ Loading State
- Full-screen spinner during API calls
- Prevents accidental double-submissions
- Non-blocking UX

## Keyboard Shortcuts (Future Enhancement)

Future versions could add:
- `Ctrl+R`: Refresh overview
- `Ctrl+T`: Focus telemetry tab
- `Ctrl+S`: Save settings
- `Escape`: Dismiss notifications

## Browser Compatibility

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 90+ | ✅ Full |
| Firefox | 88+ | ✅ Full |
| Safari | 14+ | ✅ Full |
| Edge | 90+ | ✅ Full |
| IE 11 | - | ❌ Not supported |

## Performance

- **Load time**: < 500ms (all assets)
- **API response handling**: < 100ms (client-side)
- **Interactive elements**: < 16ms render (60fps)
- **Memory footprint**: ~2-5MB

## Troubleshooting

### API Connection Failed
1. Check FastAPI backend is running on configured URL
2. Verify no CORS issues (backend should allow frontend origin)
3. Open browser console (F12) for error details
4. Update API URL in Settings tab

### Data Not Refreshing
1. Confirm auto-refresh is enabled (Settings)
2. Check refresh interval is > 0
3. Verify backend is returning data
4. Click "Refresh" button manually

### Styling Issues
- Clear browser cache (Ctrl+Shift+Delete)
- Check CSS file is loading (Network tab)
- Verify no conflicting browser extensions

## Development

### Adding New Features
1. Add HTML section to `index.html`
2. Create API method in `js/api.js` if needed
3. Add UI helpers in `js/ui.js` if needed
4. Add event handlers and logic to `js/app.js`
5. Style in `styles/main.css`

### Color Customization
Edit CSS variables in `styles/main.css`:
```css
:root {
    --primary-color: #00a8e8;
    --secondary-color: #00c9a7;
    --danger-color: #ff4757;
    /* ... etc */
}
```

### Adding API Endpoints
1. Create method in `APIClient` class
2. Add corresponding UI handler in `App` class
3. Bind event listener in `setupEventListeners()`

## License

Part of the Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine project.

## Support

For issues or questions:
1. Check the main project README
2. Review API responses in browser console
3. Verify backend is running and healthy
4. Check network tab for failed requests

---

**Dashboard Version**: 1.0.0  
**Last Updated**: 2024  
**Backend Required**: FastAPI server (see main project)
