"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class TelemetryData(BaseModel):
    """Telemetry data schema."""
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    battery_voltage: float = Field(gt=0, description="Battery voltage in volts")
    bus_voltage: float = Field(gt=0, description="Bus voltage in volts")
    wheel_rpm: float = Field(ge=0, description="Wheel RPM")
    current: float = Field(ge=0, description="Current in amperes")
    wheel_temperature: float = Field(description="Wheel temperature in celsius")

    class Config:
        json_schema_extra = {
            "example": {
                "battery_voltage": 12.5,
                "bus_voltage": 5.0,
                "wheel_rpm": 150.0,
                "current": 2.3,
                "wheel_temperature": 45.2
            }
        }


class AnomalyResult(BaseModel):
    """Anomaly detection result."""
    timestamp: datetime
    is_anomaly: bool
    confidence: float = Field(ge=0.0, le=1.0)
    anomaly_score: float
    detection_method: str = "isolation_forest"
    model_results: Optional[Dict[str, Any]] = None
    model_usage: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-04-15T18:16:14.860Z",
                "is_anomaly": True,
                "confidence": 1.0,
                "anomaly_score": 1.0,
                "detection_method": "ensemble (Isolation Forest + LOF + LSTM + Autoencoder)",
                "model_results": {
                    "isolation_forest": {
                        "is_anomaly": True,
                        "confidence": 0.91,
                        "source": "custom",
                        "using_custom_model": True,
                    },
                    "lstm": {
                        "is_anomaly": False,
                        "confidence": 0.0,
                        "source": "unavailable",
                        "using_custom_model": False,
                    },
                    "autoencoder": {
                        "is_anomaly": False,
                        "confidence": 0.0,
                        "source": "unavailable",
                        "using_custom_model": False,
                    },
                },
                "model_usage": {
                    "lstm": {
                        "source": "unavailable",
                        "using_custom_model": False,
                        "model_path": None,
                    },
                    "autoencoder": {
                        "source": "custom",
                        "using_custom_model": True,
                        "model_path": "models/autoencoder_model.keras",
                    },
                },
            }
        }


class ForecastResult(BaseModel):
    """Forecast prediction result."""
    timestamp: datetime
    feature: str
    forecast_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: bool
    database_connected: bool
    prophet_fitted: bool


class StatisticsResponse(BaseModel):
    """Statistics response."""
    total_records: int
    anomaly_count: int
    anomaly_percentage: float
    latest_timestamp: Optional[datetime] = None


class ForecasterStats(BaseModel):
    """Forecaster statistics."""
    is_fitted: bool
    feature: str
    last_training_data_points: int
