# 🚆 Rail Drishti — Indian Railways Intelligence System

> **"Drishti" (दृष्टि)** means *Vision* in Hindi. Rail Drishti gives you intelligent, data-driven vision into India's railway network.

---

## 📌 Project Overview

Rail Drishti is a **full-stack AI-powered railway intelligence platform** built on top of real Indian Railways data. It provides two distinct dashboards — one for **Passengers** and one for **Railway Admins** — each powered by Machine Learning models running on a FastAPI backend, with a sleek dark-themed frontend served via Vercel.

The system ingests 4 real datasets (stations, schedules, train details, historical delay data), merges them into a unified master dataset, trains Random Forest models in the background on startup, and exposes a clean REST API consumed by the frontend.

---

## 🗂️ Project Structure

```
databrick-main/
├── backend/
│   ├── app.py              # FastAPI server — all API routes
│   ├── models.py           # ML models: DelayPredictor, CongestionClassifier + analytics
│   ├── data_loader.py      # Data ingestion, cleaning, and master dataset builder
│   ├── utils.py            # Terminal/ANSI display utilities
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Single-page app (Passenger + Admin tabs)
│   └── config.js           # API base URL config
├── data/
│   ├── stations.json           # GeoJSON — Indian railway stations with coords/zone/state
│   ├── schedules.json          # Train schedules per station
│   ├── Train_details_22122017.csv  # Train routes, stop sequences, timings
│   └── train_delay_data_rich.csv   # Historical delay data (ML training set)
├── Dockerfile
├── docker-compose.yml
└── railway.json            # Railway.app deployment config
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, FastAPI, Uvicorn |
| ML | scikit-learn (RandomForestRegressor, RandomForestClassifier) |
| Data | Pandas, NumPy, joblib |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Deployment | Railway (backend), Vercel (frontend), Docker |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Run Locally

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 3. Open frontend
open frontend/index.html
# OR set VITE_API_URL in frontend/config.js and serve via any static server
```

### Docker

```bash
docker-compose up --build
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PORT` | Server port | `8000` |
| `RAILWAYS_DATA_DIR` | Path to data folder | `./data` |
| `FRONTEND_URL` | Allowed CORS origin for your Vercel frontend | _(none)_ |

---

## 🤖 Machine Learning Models

### 1. `DelayPredictor` — Random Forest Regressor
Predicts **arrival delay in minutes** given route features.

**Input features:**
- `distance_km` — distance between stations (clamped to 0–955 km)
- `weather` — Clear / Rainy / Foggy / Stormy / Hazy
- `day_of_week` — Monday to Sunday
- `time_of_day` — Morning / Afternoon / Evening / Night / Late Night
- `train_type` — Rajdhani / Shatabdi / Duronto / Superfast / Express / Local / Passenger

**Output:** Delay in minutes, scaled per train type:
- Rajdhani/Shatabdi: capped at 25–30 min
- Duronto: capped at 45 min
- Express/Superfast: capped at 75–90 min
- Local/Passenger: capped at 50–60 min

**Config:** 50 estimators, max_depth=10, min_samples_leaf=5

---

### 2. `CongestionClassifier` — Random Forest Classifier
Classifies a route as **Low / Medium / High congestion**.

Same 5 input features as DelayPredictor.
Also supports `predict_proba()` to return probability breakdown per class.

**Config:** 50 estimators, max_depth=8

---

## 🎫 Passenger Dashboard Features

### U1 — Train Search
Enter any train number to fetch complete intelligence for that train:
- Train name, type, source, destination, distance
- Predicted delay (ML-powered, in minutes)
- Congestion level of the route (Low / Medium / High)
- Risk level (Low / Medium / High)
- Reliability score (0–100)
- Current weather context
- Time of day bucket

### U2 — Delay Reasoning Engine
After predicting delay, the system generates **human-readable reasons** for it:
- 🌫️ Foggy conditions → reduced speed operations active
- 🌧️ Rainfall → track adhesion and signal visibility affected
- ⛈️ Storm advisory → speed restrictions imposed
- 🔴 Route heavily congested → multiple trains sharing track
- ⏱️ Peak hour (Morning/Evening) → signal queuing likely
- ⚠️ Cascading upstream delay
- 🚨 Severe delay → possible track maintenance

### U3 — Reliability Score
A 0–100 score calculated from:
- Number of stops (complexity penalty)
- Average route distance (exposure penalty)
- Median historical delay (delay penalty)

Displayed as both a number and a visual progress bar.

### U4 — Stop-by-Stop Schedule with Estimated Arrival
For each stop on the route (up to 15 shown):
- Station code + name
- Scheduled arrival and departure times
- **Estimated arrival** = scheduled time + predicted delay minutes
- Distance in km
- Platform number (where available)

### U5 — Alternative Trains on Same Route
Shows other trains that share at least 2 stations with the searched train — useful as alternatives when the main train is delayed.

---

## 🏛️ Admin Control Dashboard Features

### A0 — Operational Dashboard (Summary KPIs)
Quick-load overview:
- Total trains in network
- Total stations
- Total route entries
- Zone-wise train distribution

### A1 — Most Congested Stations
Top-N stations ranked by **congestion score**:

```
congestion_score = trains_stopping × log(1 + routes_through)
```

Displays station code, name, zone, state, train count, route count, visual gauge bar, and Low/Medium/High badge.

### A2 — Congested Rail Corridors
Top-N route corridors (Source → Destination) ranked by **route score**:

```
route_score = trains_on_corridor × log(1 + stations_on_corridor)
```

Includes corridor name, train count, station count, max distance, and load gauge.

### A3 — Congestion Risk Forecasting (Multi-Window)
Predicts congestion risk across **3 time windows**:
- **Now** — current conditions
- **+2 Hours** — near-future
- **Tomorrow AM** — next-day morning

For each window, runs 4 scenario predictions (Express/Superfast/Local/Foggy-Express) with predicted delay and congestion level.

### A4 — Cascading Delay Simulation
Given a train number, simulates how its delay propagates to downstream trains:

- Analyses top route corridors for that train
- Predicts base delay per corridor (ML model)
- Applies a **1.4× cascade factor** (each downstream stop picks up ~20–40% more delay)
- Classifies cascade risk as Low / Medium / High
- Returns: input train, affected trains, station hubs, delay passed in minutes

### A5 — Rerouting Options
For a given train, finds alternative trains sharing **≥ 2 stations** with it, ranked by shared station count. Returns top 5 with train number, name, route, and type.

### A6 — Priority Clearance Queue
Lists trains sorted by operational priority:
- **P1** — Rajdhani, Shatabdi, Duronto (highest priority)
- **P2** — Superfast
- **P3** — Express
- **P4** — Local, Passenger

Used to decide which trains get signal/track clearance first in a congestion scenario.

### A7 — Zone / Division Summary
Shows all Indian railway zones with:
- Number of stations
- Number of trains
- Number of states covered
- Overall congestion level (Low / Medium / High)

### A8 — Network Bottleneck Detection
Automatically identifies stations with extreme congestion:
- Flags stations above the **90th percentile** congestion score threshold
- Classifies them as: High / Critical / Extreme
- Provides reason: "Congestion score above 90th percentile threshold"

---

## 🌐 API Reference

### System
| Endpoint | Description |
|---|---|
| `GET /` | Health ping |
| `GET /health` | Full status: loaded, train count, station count |

### Passenger (User) APIs
| Endpoint | Params | Description |
|---|---|---|
| `GET /api/user/train-info` | `train_no` | Full train intelligence |
| `GET /api/user/alternatives` | `train_no` | Alternative trains on same route |
| `GET /api/user/station-info` | `station_code` | Station metadata |
| `GET /api/user/nearby-stations` | `station_code`, `radius_km` | Haversine-based nearby stations |

### Admin APIs
| Endpoint | Params | Description |
|---|---|---|
| `GET /api/admin/congested-stations` | `top_n` | Congestion-ranked stations |
| `GET /api/admin/congested-corridors` | `top_n` | Congestion-ranked corridors |
| `GET /api/admin/congestion-risk` | — | Multi-window risk forecast |
| `GET /api/admin/cascading-delays` | `train_no` | Cascade simulation |
| `GET /api/admin/rerouting` | `train_no` | Rerouting options |
| `GET /api/admin/zone-summary` | — | Zone-level operational summary |
| `GET /api/admin/bottlenecks` | — | Network bottleneck stations |
| `GET /api/admin/priority-clearance` | `top_n` | Priority-sorted train queue |
| `GET /api/admin/operational-dashboard` | — | Full KPI summary |

---

## 📊 Data Sources

| File | Records | Description |
|---|---|---|
| `stations.json` | ~8,000+ stations | GeoJSON with lat/lon, zone, state |
| `schedules.json` | ~100,000+ entries | Arrival/departure times per stop |
| `Train_details_22122017.csv` | ~200,000+ rows | Route sequences with distances & timings |
| `train_delay_data_rich.csv` | ~10,000+ rows | Historical delay data (ML training set) |

---

## 🎨 Frontend Design System

The UI uses a dark-mode design with custom CSS variables:

| Token | Value | Use |
|---|---|---|
| `--bg-base` | `#05090f` | Page background |
| `--accent-blue` | `#1a8fff` | Primary actions |
| `--accent-cyan` | `#00d4ff` | Highlights |
| `--accent-amber` | `#f59e0b` | Warnings |
| `--accent-red` | `#ef4444` | Danger/high delay |
| `--accent-green` | `#22c55e` | OK/low delay |
| `--font-display` | Bebas Neue | Headers |
| `--font-mono` | Space Mono | Codes/data |
| `--font-body` | DM Sans | Body text |

Additional UX features: noise texture overlay, animated pulse status dot, live clock ticker, gradient glows on cards, sticky header with tab switching.

---

## 🚢 Deployment

### Backend → Railway.app
```json
// railway.json
{ "build": { "builder": "DOCKERFILE" }, "deploy": { "startCommand": "uvicorn backend.app:app ..." } }
```

### Frontend → Vercel
```json
// vercel.json
{ "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }
```

Set `FRONTEND_URL` env var on Railway to your Vercel deployment URL to enable CORS.

---

## 📄 License

This project uses publicly available Indian Railways data. Built for educational and research purposes.

---

*Rail Drishti — See the railway, understand it, optimise it.*
