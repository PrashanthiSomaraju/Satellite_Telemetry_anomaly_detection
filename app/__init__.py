"""Initialize app package."""

__version__ = "2.0.0"
__description__ = "Satellite Telemetry Analysis System"

from app.config import settings
from app.database import db_manager
from app.models import model_manager
from app.forecast import forecast_manager
from app.inference import AnomalyDetector

__all__ = [
    "settings",
    "db_manager",
    "model_manager",
    "forecast_manager",
    "AnomalyDetector"
]
