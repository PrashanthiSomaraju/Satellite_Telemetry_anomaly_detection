# 🛰️ Satellite Telemetry Analysis System

A real-time satellite telemetry monitoring system for **anomaly detection, time-series forecasting, and telemetry analysis**.

The system uses **Isolation Forest** to identify abnormal telemetry readings and **Prophet** to forecast future telemetry trends. A **FastAPI backend** provides REST APIs, while **MongoDB** stores telemetry and anomaly data. A lightweight dashboard provides real-time monitoring and visualization.

---

## ✨ Features

* 🔍 **Real-time anomaly detection** using Isolation Forest
* 📈 **Time-series forecasting** using Prophet
* 🗄️ **MongoDB persistence** for telemetry and anomaly records
* ⚡ **FastAPI REST API** with automatic API documentation
* 📊 **Web dashboard** for telemetry monitoring
* 🔄 **Recent telemetry and anomaly history**
* 🧠 **Model-based inference** using pre-trained models
* 💻 **CPU-based execution** — no GPU required

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Satellite / Data  │
                    │       Source        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Backend  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │  Isolation  │  │   Prophet   │  │  MongoDB    │
       │   Forest    │  │ Forecasting │  │  Database   │
       └──────┬──────┘  └──────┬──────┘  └─────────────┘
              │                │
              └────────┬───────┘
                       ▼
              ┌──────────────────┐
              │ Monitoring       │
              │ Dashboard        │
              └──────────────────┘
```

### Processing Flow

1. Telemetry data is submitted to the FastAPI backend.
2. The backend validates the incoming telemetry.
3. Isolation Forest analyzes the telemetry for potential anomalies.
4. Results are stored in MongoDB.
5. Prophet generates future telemetry forecasts when requested.
6. The dashboard retrieves telemetry, anomalies, and forecasts through REST APIs.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure MongoDB

Create a `.env` file based on `.env.example`:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-host>/satellite_telemetry?retryWrites=true&w=majority
DB_NAME=satellite_telemetry
LOG_LEVEL=INFO
```

Make sure your MongoDB Atlas cluster allows connections from your machine.

### 4. Start the Backend

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

### 5. Open the Dashboard

Open:

```text
frontend/index.html
```

in your browser.

### 6. API Documentation

FastAPI automatically provides interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 📁 Project Structure

```text
satellite-telemetry-analysis/
│
├── app/
│   ├── main.py          # FastAPI application and routes
│   ├── database.py      # MongoDB connection and operations
│   ├── models.py        # ML model loading
│   ├── forecast.py      # Prophet forecasting
│   ├── inference.py     # Anomaly detection inference
│   ├── schemas.py       # Request/response validation
│   └── config.py        # Application configuration
│
├── frontend/
│   └── index.html       # Monitoring dashboard
│
├── models/
│   ├── isolation_forest.pkl
│   └── prophet_model.pkl
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint                | Description                                    |
| ------ | ----------------------- | ---------------------------------------------- |
| `POST` | `/api/telemetry`        | Submit telemetry and perform anomaly detection |
| `GET`  | `/api/telemetry/recent` | Retrieve recent telemetry records              |
| `GET`  | `/api/anomalies`        | Retrieve detected anomalies                    |
| `POST` | `/api/forecast`         | Generate a 24-hour telemetry forecast          |
| `POST` | `/api/train`            | Train/update the Prophet forecasting model     |
| `GET`  | `/health`               | Check backend and system health                |

---

## 📡 Telemetry Data Format

### Example Request

```json
{
  "battery_voltage": 12.5,
  "bus_voltage": 5.0,
  "wheel_rpm": 150.0,
  "current": 2.3,
  "wheel_temperature": 45.2
}
```

### Telemetry Parameters

| Parameter           | Description               |
| ------------------- | ------------------------- |
| `battery_voltage`   | Satellite battery voltage |
| `bus_voltage`       | Electrical bus voltage    |
| `wheel_rpm`         | Wheel rotational speed    |
| `current`           | Current measurement       |
| `wheel_temperature` | Wheel temperature         |

---

## 🤖 Machine Learning Models

### Isolation Forest

Isolation Forest is used for **unsupervised anomaly detection**.

It identifies telemetry observations that differ significantly from normal operating patterns.

```text
Telemetry
    │
    ▼
Feature Extraction
    │
    ▼
Isolation Forest
    │
    ├── Normal
    │
    └── Anomaly
```

### Prophet

Prophet is used for **time-series forecasting**.

Historical telemetry data is used to estimate future values and identify expected trends.

```text
Historical Telemetry
        │
        ▼
   Prophet Model
        │
        ▼
 Future Forecast
```

---

## 🗄️ Database

The system uses **MongoDB** to persist:

* Telemetry measurements
* Anomaly detection results
* Timestamps
* Model inference results
* Forecast-related data

MongoDB provides flexible storage for continuously arriving telemetry records.

---

## ⚙️ Environment Variables

Create a `.env` file:

```env
MONGODB_URI=your-mongodb-connection-string
DB_NAME=satellite_telemetry
LOG_LEVEL=INFO
```

> ⚠️ Never commit `.env` or database credentials to Git. Use `.env.example` for sharing configuration templates.

---

## 🛠️ Troubleshooting

### MongoDB Connection Error

Check:

* Your `MONGODB_URI`
* MongoDB Atlas network access settings
* Database credentials
* Internet connectivity

### Port Already in Use

Run the backend on another port:

```bash
python -m uvicorn app.main:app --port 8001
```

### Prophet Model Not Found

Make sure the trained model exists at:

```text
models/prophet_model.pkl
```

Then restart the backend.

### Isolation Forest Model Not Found

Make sure the trained model exists at:

```text
models/isolation_forest.pkl
```

---

## 📋 Requirements

* Python 3.9+
* MongoDB / MongoDB Atlas
* FastAPI
* Uvicorn
* Prophet
* scikit-learn
* MongoDB Python driver
* Modern web browser

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

* Real-time telemetry streaming using WebSockets
* Additional anomaly detection models
* Multi-parameter correlation analysis
* Alert and notification system
* Historical telemetry visualization
* Model performance monitoring
* Containerized deployment with Docker
* Authentication and role-based access control

---

## 📄 License

This project is intended for educational and research purposes.
