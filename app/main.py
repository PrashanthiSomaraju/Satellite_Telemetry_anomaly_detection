"""FastAPI application for satellite telemetry anomaly detection."""

import logging
import os
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import (
    TelemetryData,
    AnomalyResult,
    HealthStatus,
    StatisticsResponse
)
from app.database import db_manager
from app.models import model_manager
from app.forecast import forecast_manager
from app.inference import AnomalyDetector

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
anomaly_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("🚀 Application starting up...")

    # Connect to MongoDB
    db_connected = db_manager.connect()
    if not db_connected:
        logger.warning("⚠️  MongoDB connection failed - proceeding with degraded mode")

    # Load models
    model_manager.load_models()

    # Initialize anomaly detector
    global anomaly_detector
    anomaly_detector = AnomalyDetector(model_manager, settings)

    # Load Prophet model if available
    forecast_manager.load_model()

    logger.info("✓ Application started successfully")

    yield

    # Shutdown
    logger.info("🛑 Application shutting down...")
    db_manager.close()
    logger.info("✓ Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Satellite Telemetry Analysis",
    description="Real-time anomaly detection and forecasting for satellite telemetry",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ============ Health & Status Endpoints ============


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        models_loaded=model_manager.isolation_forest is not None,
        database_connected=db_manager.db is not None,
        prophet_fitted=forecast_manager.is_fitted
    )


@app.get("/api/stats", response_model=StatisticsResponse)
async def get_statistics():
    """Get system statistics."""
    try:
        telemetry = db_manager.get_recent_telemetry(limit=10000)
        anomalies = db_manager.get_recent_anomalies(limit=10000)

        total = len(telemetry)
        anomaly_count = len(anomalies) if anomalies else 0
        anomaly_percentage = (anomaly_count / total * 100) if total > 0 else 0

        latest_timestamp = None
        if telemetry:
            latest_timestamp = telemetry[0].get("timestamp")

        return StatisticsResponse(
            total_records=total,
            anomaly_count=anomaly_count,
            anomaly_percentage=round(anomaly_percentage, 2),
            latest_timestamp=latest_timestamp
        )
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Telemetry Endpoints ============


@app.post("/api/telemetry", response_model=AnomalyResult)
async def ingest_telemetry(data: TelemetryData):
    """Ingest telemetry and detect anomalies using ensemble methods."""
    if anomaly_detector is None:
        raise HTTPException(status_code=500, detail="Anomaly detector not initialized")

    try:
        # Detect anomaly using ensemble (Isolation Forest + LSTM)
        is_anomaly, confidence, detection_details = anomaly_detector.detect(data.dict())

        # Insert telemetry record
        telem_dict = data.dict()
        db_manager.insert_telemetry(telem_dict)

        # Insert anomaly record if anomalous
        if is_anomaly:
            anomaly_dict = {
                "timestamp": data.timestamp or datetime.utcnow(),
                "is_anomaly": True,
                "confidence": confidence,
                "telemetry_data": telem_dict,
                "detection_details": detection_details
            }
            db_manager.insert_anomaly(anomaly_dict)

        return AnomalyResult(
            timestamp=data.timestamp or datetime.utcnow(),
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_score=confidence,
            detection_method="ensemble (Isolation Forest + LOF + LSTM + Autoencoder)",
            model_results={
                "isolation_forest": detection_details.get("isolation_forest"),
                "local_outlier_factor": detection_details.get("local_outlier_factor"),
                "lstm": detection_details.get("lstm"),
                "autoencoder": detection_details.get("autoencoder"),
                "hard_limit": detection_details.get("hard_limit"),
            },
            model_usage=model_manager.get_model_usage_summary()
        )
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/telemetry/recent", response_model=dict)
async def get_recent_telemetry(limit: int = 100):
    """Get recent telemetry records."""
    try:
        data = db_manager.get_recent_telemetry(limit=limit)
        return {"count": len(data), "data": data}
    except Exception as e:
        logger.error(f"Error fetching telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomalies", response_model=dict)
async def get_anomalies(limit: int = 50):
    """Get detected anomalies."""
    try:
        data = db_manager.get_recent_anomalies(limit=limit)
        return {"count": len(data), "data": data}
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Forecast Endpoints ============


@app.post("/api/forecast")
async def generate_forecast(feature: str = "bus_voltage", periods: int = 24):
    """Generate forecast for specified feature."""
    try:
        if not forecast_manager.is_fitted:
            raise HTTPException(status_code=400, detail="Prophet models not trained yet")

        available_features = forecast_manager.get_available_features()
        if feature not in available_features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{feature}' not available. Available: {available_features}"
            )

        forecast_df = forecast_manager.forecast(feature=feature, periods=periods)
        if forecast_df is None or len(forecast_df) == 0:
            raise HTTPException(status_code=500, detail=f"Failed to generate forecast for '{feature}'")

        # Convert to JSON-serializable format
        results = []
        for _, row in forecast_df.iterrows():
            results.append({
                "timestamp": row["ds"].isoformat(),
                "forecast": float(row["yhat"]),
                "lower_bound": float(row["yhat_lower"]),
                "upper_bound": float(row["yhat_upper"])
            })

        return {"feature": feature, "periods": periods, "data": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/status")
async def forecast_status():
    """Get forecast models status."""
    return {
        "is_fitted": forecast_manager.is_fitted,
        "available_features": forecast_manager.get_available_features(),
        "model_type": "Prophet (Multi-feature support)"
    }


# ============ Training Endpoints (for backend use) ============


@app.post("/api/train")
async def train_models(feature: str = None):
    """Train Prophet model(s) on existing telemetry data.

    Args:
        feature: Specific feature to train. If None, trains all supported features.
    """
    try:
        # Get training data
        training_data = db_manager.get_telemetry_for_training(limit=500)

        if len(training_data) < 20:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for training. Need at least 20 samples, got {len(training_data)}"
            )

        logger.info(f"Training Prophet model(s) with {len(training_data)} samples...")

        trained_features = []

        if feature:
            # Train specific feature
            success = forecast_manager.train(training_data, feature_name=feature)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to train Prophet model for '{feature}'")
            forecast_manager.save_model(feature)
            trained_features.append(feature)
        else:
            # Train all supported features
            for feat in forecast_manager.supported_features:
                try:
                    success = forecast_manager.train(training_data, feature_name=feat)
                    if success:
                        forecast_manager.save_model(feat)
                        trained_features.append(feat)
                        logger.info(f"Successfully trained model for '{feat}'")
                    else:
                        logger.warning(f"Failed to train model for '{feat}'")
                except Exception as e:
                    logger.warning(f"Error training model for '{feat}': {e}")

        if not trained_features:
            raise HTTPException(status_code=500, detail="Failed to train any Prophet models")

        return {
            "status": "success",
            "message": f"Prophet models trained successfully for: {trained_features}",
            "trained_features": trained_features,
            "samples_used": len(training_data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomaly-forecast")
async def get_anomaly_forecast(feature: str = "bus_voltage", periods: int = 24):
    """
    Get forecast data to predict when anomalies may happen.

    Args:
        feature: Feature to forecast (e.g., battery_temp, bus_voltage, wheel_rpm)
        periods: Number of periods to forecast
    """
    try:
        available_features = forecast_manager.get_available_features()

        if not available_features:
            raise HTTPException(
                status_code=400,
                detail="No Prophet models trained. Run /api/train first."
            )

        if feature not in available_features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{feature}' not available. Use: {available_features}"
            )

        # Get forecast
        forecast_df = forecast_manager.forecast(feature=feature, periods=periods)

        if forecast_df is None or len(forecast_df) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate forecast")

        # Get recent data for context
        recent_data = db_manager.get_recent_telemetry(limit=50)

        # Convert forecast to JSON
        results = []
        for _, row in forecast_df.iterrows():
            results.append({
                "timestamp": row["ds"].isoformat(),
                "forecast": float(row["yhat"]),
                "lower_bound": float(row["yhat_lower"]),
                "upper_bound": float(row["yhat_upper"]),
                "confidence_interval": f"[{float(row['yhat_lower']):.2f}, {float(row['yhat_upper']):.2f}]"
            })

        return {
            "feature": feature,
            "periods": periods,
            "forecast_data": results,
            "recent_values": [item.get(feature, 0) for item in recent_data[-10:]]  # Last 10 values
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating anomaly forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Serve frontend HTML."""
    from fastapi.responses import FileResponse
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {
        "message": "Satellite Telemetry Analysis System v2.0",
        "features": [
            "Anomaly Detection (Isolation Forest + Local Outlier Factor + Autoencoder)",
            "Time Series Forecasting (Prophet)",
            "MongoDB Data Storage"
        ],
        "endpoints": {
            "health": "/health",
            "api": "/api/telemetry, /api/anomalies, /api/forecast",
            "docs": "/docs",
            "train": "/api/train (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port
    )
