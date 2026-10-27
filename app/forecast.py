"""Prophet-based forecasting manager with multi-feature support."""

import os
import pickle
import logging
from typing import Optional, List, Dict, Any
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    logger.warning("Prophet not installed. Install with: pip install prophet")


class ProphetForecastManager:
    """Prophet-based time series forecasting with support for multiple features."""

    def __init__(self):
        self.prophet_models: Dict[str, Any] = {}  # Feature-specific models
        self.models_path = os.path.join(os.path.dirname(__file__), "..", "models")
        self.is_fitted = False
        self.supported_features = [
            "battery_temp",
            "bus_current",
            "bus_voltage",
            "wheel_rpm",
            "wheel_temp"
        ]

    def load_model(self, feature_name: Optional[str] = None) -> bool:
        """
        Load Prophet models from disk.

        Args:
            feature_name: Specific feature to load. If None, loads all available.
        """
        try:
            os.makedirs(self.models_path, exist_ok=True)

            if feature_name:
                # Load specific model
                return self._load_single_model(feature_name)
            else:
                # Load all available models
                return self._load_all_models()

        except Exception as e:
            logger.error(f"Error loading Prophet models: {e}")
            return False

    def _load_single_model(self, feature_name: str) -> bool:
        """Load a single Prophet model for a specific feature."""
        try:
            load_path = os.path.join(self.models_path, f"prophet_{feature_name}.pkl")

            if not os.path.exists(load_path):
                logger.warning(f"Prophet model not found for feature '{feature_name}' at {load_path}")
                return False

            with open(load_path, "rb") as f:
                model = pickle.load(f)

            self.prophet_models[feature_name] = model
            logger.info(f"✓ Loaded Prophet model for '{feature_name}'")
            self._update_fitted_status()
            return True

        except Exception as e:
            logger.error(f"Error loading Prophet model for '{feature_name}': {e}")
            return False

    def _load_all_models(self) -> bool:
        """Load all available Prophet models from the models directory."""
        try:
            models_found = 0

            if os.path.exists(self.models_path):
                for filename in os.listdir(self.models_path):
                    if filename.startswith("prophet_") and filename.endswith(".pkl"):
                        # Extract feature name
                        feature_name = filename.replace("prophet_", "").replace(".pkl", "")
                        filepath = os.path.join(self.models_path, filename)

                        try:
                            with open(filepath, "rb") as f:
                                model = pickle.load(f)
                            self.prophet_models[feature_name] = model
                            models_found += 1
                            logger.info(f"✓ Loaded Prophet model for '{feature_name}'")
                        except Exception as e:
                            logger.warning(f"Failed to load {filename}: {e}")

            if models_found > 0:
                logger.info(f"Loaded {models_found} Prophet models: {list(self.prophet_models.keys())}")
                self._update_fitted_status()
                return True
            else:
                logger.warning("No Prophet models found in models directory")
                return False

        except Exception as e:
            logger.error(f"Error loading all Prophet models: {e}")
            return False

    def _update_fitted_status(self) -> None:
        """Update is_fitted status based on loaded models."""
        self.is_fitted = len(self.prophet_models) > 0

    def forecast(self, feature_name: str, periods: int = 24) -> Optional[pd.DataFrame]:
        """
        Generate forecast for specified feature and periods.

        Args:
            feature_name: Feature to forecast
            periods: Number of periods to forecast

        Returns:
            DataFrame with forecast data or None if model not available
        """
        try:
            model = self.prophet_models.get(feature_name)

            if model is None:
                logger.warning(f"No Prophet model available for '{feature_name}'")
                return None

            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq="H")
            forecast = model.predict(future)

            return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

        except Exception as e:
            logger.error(f"Error generating forecast for '{feature_name}': {e}")
            return None

    def get_next_value(self, feature_name: str) -> Optional[float]:
        """
        Get next predicted value for a feature.

        Args:
            feature_name: Feature to get prediction for

        Returns:
            Predicted value or None
        """
        try:
            forecast = self.forecast(feature_name, periods=1)

            if forecast is not None and len(forecast) > 0:
                return float(forecast.iloc[-1]["yhat"])
            return None

        except Exception as e:
            logger.error(f"Error getting next value for '{feature_name}': {e}")
            return None

    def get_available_features(self) -> List[str]:
        """Get list of features with loaded Prophet models."""
        return list(self.prophet_models.keys())

    def train(self, data: List[Dict[str, Any]], feature_name: str) -> bool:
        """
        Train Prophet model on telemetry data for a specific feature.

        Args:
            data: List of telemetry records
            feature_name: Feature to train on
        """
        try:
            if len(data) < 20:
                logger.warning(f"Insufficient data for Prophet training: {len(data)} < 20")
                return False

            if Prophet is None:
                logger.error("Prophet library not available")
                return False

            # Prepare data for Prophet
            df = pd.DataFrame(data)
            if "timestamp" not in df.columns or feature_name not in df.columns:
                logger.error(f"Data must contain 'timestamp' and '{feature_name}' columns")
                return False

            # Prophet expects columns named 'ds' (timestamp) and 'y' (target)
            df_prophet = pd.DataFrame({
                "ds": pd.to_datetime(df["timestamp"]),
                "y": df[feature_name].astype(float)
            })

            # Remove any NaN values
            df_prophet = df_prophet.dropna()

            if len(df_prophet) < 20:
                logger.warning(f"Insufficient data after cleaning: {len(df_prophet)}")
                return False

            logger.info(f"Training Prophet model for '{feature_name}' with {len(df_prophet)} samples...")

            # Create and train Prophet model
            prophet_model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=True,
                interval_width=0.95,
                seasonality_mode="additive"
            )

            prophet_model.fit(df_prophet)
            self.prophet_models[feature_name] = prophet_model
            self._update_fitted_status()

            logger.info(f"✓ Prophet model trained successfully for '{feature_name}'")
            return True

        except Exception as e:
            logger.error(f"Error training Prophet model for '{feature_name}': {e}")
            return False

    def save_model(self, feature_name: str, path: Optional[str] = None) -> bool:
        """
        Save Prophet model to disk.

        Args:
            feature_name: Feature whose model to save
            path: Custom path. If None, uses default naming
        """
        try:
            model = self.prophet_models.get(feature_name)

            if model is None:
                logger.error(f"No Prophet model available for '{feature_name}'")
                return False

            save_path = path or os.path.join(self.models_path, f"prophet_{feature_name}.pkl")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"✓ Prophet model for '{feature_name}' saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving Prophet model for '{feature_name}': {e}")
            return False


# Global instance
forecast_manager = ProphetForecastManager()
