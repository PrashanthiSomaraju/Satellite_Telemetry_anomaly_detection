# Quick Start Guide

## Prerequisites
- Python 3.10+
- MongoDB Atlas account (free tier works)

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure MongoDB
Create `.env` file in the project root:
```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-host>/satellite_telemetry?retryWrites=true&w=majority
DB_NAME=satellite_telemetry
LOG_LEVEL=INFO
```

**Get your URI from [MongoDB Atlas](https://www.mongodb.com/cloud/atlas):**
1. Create a free cluster → Connect with URI → Copy connection string
2. Replace `username` and `password` with your credentials

### 3. Add Models (Optional)
Place in `models/` directory:
- `prophet_model.pkl` - Time series forecasting
- `isolation_forest.pkl` - Anomaly detection

*Models auto-generate if missing (good for testing)*

### 4. Start Backend
```bash
python -m uvicorn app.main:app --reload
```
Backend available at: `http://localhost:8000`

### 5. Open Frontend
Open this file in your browser: `frontend/index.html`

## Test It

### Via Frontend
1. Open `frontend/index.html`
2. Enter test data (Battery: 12.5, Bus: 5.0, RPM: 150, Current: 2.3, Temp: 45.2)
3. Click "Detect Anomalies"
4. See results instantly

### Via API
```bash
# Submit telemetry
curl -X POST http://localhost:8000/api/telemetry \
  -H "Content-Type: application/json" \
  -d '{"battery_voltage": 12.5, "bus_voltage": 5.0, "wheel_rpm": 150, "current": 2.3, "wheel_temperature": 45.2}'

# View data
curl http://localhost:8000/api/telemetry/recent

# View anomalies
curl http://localhost:8000/api/anomalies

# Generate forecast
curl -X POST http://localhost:8000/api/forecast

# Health check
curl http://localhost:8000/health
```

## API Documentation
Interactive Swagger docs: `http://localhost:8000/docs`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MongoDB fails | Check `.env` URI, verify network access in MongoDB Atlas |
| Port 8000 in use | `python -m uvicorn app.main:app --port 8001` |
| No model files | System creates mock models (fine for testing) |
| No data shows | Submit at least 1 telemetry record first |
| CORS error | Already enabled - frontend works from any location |

## Project Files

```
app/               → Backend API
  ├── main.py        - FastAPI endpoints
  ├── database.py    - MongoDB operations
  ├── forecast.py    - Prophet time series
  ├── inference.py   - Anomaly detection
  ├── models.py      - Model loading
  ├── schemas.py     - Data validation
  └── config.py      - Settings

frontend/         → React dashboard
  └── index.html     - Single-file app

models/           → Model files (.pkl)
```

## Key Features

✅ Real-time anomaly detection
✅ Time series forecasting
✅ Live React dashboard
✅ MongoDB persistence
✅ No GPU needed
✅ Production-ready FastAPI

## Next Steps

1. Load historical telemetry into MongoDB
2. Train Prophet: `curl -X POST http://localhost:8000/api/train`
3. Generate forecasts and monitor anomalies
4. Integrate with your notification system

---

**Ready? Start with:** `pip install -r requirements.txt` → Create `.env` → Run backend → Open frontend
# Quick Start Guide - v2.0\n\n## ⚡ 5-Minute Setup\n\n### Step 1: Install Dependencies\n```bash\npip install -r requirements.txt\n```\n\n### Step 2: Configure MongoDB\nCreate `.env` file:\n```env\nMONGODB_URI=mongodb+srv://<username>:<password>@<cluster-host>/satellite_telemetry?retryWrites=true&w=majority```\n\n### Step 3: Start Backend\n```bash\npython -m uvicorn app.main:app --reload\n```\nAPI available at: http://localhost:8000/docs\n\n### Step 4: Open Frontend\nJust open `frontend/index.html` in your browser\n\n## 🚀 Quick API Tests\n\n### Submit Telemetry\n```bash\ncurl -X POST http://localhost:8000/api/telemetry \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\n    \"battery_voltage\": 12.5,\n    \"bus_voltage\": 5.0,\n    \"wheel_rpm\": 150,\n    \"current\": 2.3,\n    \"wheel_temperature\": 45.2\n  }'\n```\n\n### Get Recent Data\n```bash\ncurl http://localhost:8000/api/telemetry/recent?limit=10\n```\n\n### Get Anomalies\n```bash\ncurl http://localhost:8000/api/anomalies?limit=5\n```\n\n### Health Check\n```bash\ncurl http://localhost:8000/health\n```\n\n## 🔧 Train Prophet Model\n\nOption 1 - Google Colab (Recommended):\n1. Upload `Train_Prophet_Model.ipynb` to Colab\n2. Run all cells\n3. Download `prophet_model.pkl`\n4. Place in `models/` directory  \n5. Restart backend\n\nOption 2 - Locally:\n```bash\ncurl -X POST http://localhost:8000/api/train\n```\nRequires 20+ telemetry records in MongoDB.\n\n## 📁 Project Structure\n\n```\napp/              → Backend\n  ├── main.py     → FastAPI server\n  ├── database.py → MongoDB manager\n  ├── models.py   → Isolation Forest loader\n  ├── forecast.py → Prophet forecasting\n  └── inference.py→ Anomaly detection\n\nfrontend/         → React dashboard\n  └── index.html\n\nmodels/           → Trained models (.pkl files)\n```\n\n## ⚙️ Configuration\n\nEdit `app/config.py` to customize:\n```python\nanomaly_threshold = 0.7           # Detection sensitivity\nmin_data_for_training = 50        # Data needed for Prophet\n```\n\n## 🐛 Troubleshooting\n\n| Problem | Solution |\n|---------|----------|\n| MongoDB connection failed | Check `.env` URI |\n| Port 8000 in use | `python -m uvicorn app.main:app --port 8001` |\n| Prophet not trained | Submit 20+ records then call `/api/train` |\n| Frontend won't load | Use: `python -m http.server 8001 --directory frontend` |
5. Copy connection string
6. Replace in `.env`:
   ```
   mongodb+srv://<username>:<password>@<cluster-host>/satellite_telemetry?retryWrites=true&w=majority
   ```

## Environment Variables

### Required
- `MONGODB_URI` - MongoDB Atlas connection string

### Optional (defaults shown)
- `RETRAIN_INTERVAL=3600` - Retraining interval in seconds
- `SLIDING_WINDOW_SIZE=3000` - Max records for retraining
- `PORT=8000` - API port
- `HOST=0.0.0.0` - API host

## Troubleshooting

### Models Not Found
```
Error: isolation.pkl not found
```
**Solution**: Copy `isolation.pkl` and `lstm.keras` to `models/` folder

### MongoDB Connection Failed
```
Error: Failed to connect to MongoDB
```
**Solution**:
- Check `.env` file has correct `MONGODB_URI`
- Verify IP whitelist in MongoDB Atlas (add your IP)
- Test connection: `python -c "from pymongo import MongoClient; MongoClient('YOUR_URI').admin.command('PING')"`

### Port 8000 Already In Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill the process or use different port:
```bash
python -m uvicorn app.main:app --port 9000
```

### Missing Dependencies
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**:
```bash
pip install -r requirements.txt
```

## Development Tips

- Use `--reload` flag for auto-restart on file changes
- Check logs for errors: Look at console output
- Test database: Use MongoDB Compass (free GUI tool)
- Monitor memory: Use Task Manager (Windows) or `top` (Linux)

---

**That's it!** Your system is now running locally without Docker. Perfect for development and testing before EC2 deployment.
