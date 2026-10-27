"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017/satellite_telemetry"
    db_name: str = "satellite_telemetry"

    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Model Configuration
    isolation_forest_contamination: float = 0.05
    prophet_seasonality_mode: str = "additive"
    anomaly_threshold: float = 0.7
    min_data_for_training: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


settings = Settings()
