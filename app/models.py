"""Model manager for loading pre-trained models."""

import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from tensorflow import keras
except ImportError:
    keras = None
    logger.warning("TensorFlow/Keras not installed.")


class ModelManager:
    """Manage pre-trained models (Isolation Forest, LOF, LSTM, Autoencoder, Prophet)."""

    def __init__(self):
        self.isolation_forest: Optional[IsolationForest] = None
        self.local_outlier_factor: Optional[LocalOutlierFactor] = None
        self.lstm_model: Optional[Any] = None
        self.autoencoder_model: Optional[Any] = None
        self.prophet_models: Dict[str, Any] = {}  # Feature-specific Prophet models
        self.models_path = os.path.join(os.path.dirname(__file__), "..", "models")
        self.using_mock_models = False
        self.model_sources: Dict[str, str] = {
            "isolation_forest": "unavailable",
            "local_outlier_factor": "unavailable",
            "lstm": "unavailable",
            "autoencoder": "unavailable",
        }
        self.model_paths: Dict[str, Optional[str]] = {
            "isolation_forest": None,
            "local_outlier_factor": None,
            "lstm": None,
            "autoencoder": None,
        }

    def load_models(self) -> bool:
        """Load pre-trained models at startup."""
        try:
            os.makedirs(self.models_path, exist_ok=True)

            # Load Isolation Forest (check both naming conventions)
            model_names = ["isolation_model.pkl", "isolation_forest.pkl"]
            if_path = None

            for model_name in model_names:
                path = os.path.join(self.models_path, model_name)
                if os.path.exists(path):
                    if_path = path
                    break

            if if_path:
                try:
                    with open(if_path, "rb") as f:
                        self.isolation_forest = pickle.load(f)
                    self.model_sources["isolation_forest"] = "custom"
                    self.model_paths["isolation_forest"] = if_path
                    logger.info(f"✓ Isolation Forest model loaded from {os.path.basename(if_path)}")
                except Exception as e:
                    logger.warning(f"Failed to load Isolation Forest: {e}. Using mock model.")
                    self.isolation_forest = self._create_mock_isolation_forest()
                    self.model_sources["isolation_forest"] = "mock"
                    self.model_paths["isolation_forest"] = None
                    self.using_mock_models = True
            else:
                logger.warning("Isolation Forest model not found. Creating mock model.")
                self.isolation_forest = self._create_mock_isolation_forest()
                self.model_sources["isolation_forest"] = "mock"
                self.model_paths["isolation_forest"] = None
                self.using_mock_models = True

            # Load LSTM model
            self._load_lstm_model()

            # Load Local Outlier Factor model
            self._load_lof_model()

            # Load Autoencoder model
            self._load_autoencoder_model()

            # Load Prophet models for different features
            self._load_prophet_models()

            if self.using_mock_models:
                logger.warning(
                    "⚠️  Using MOCK MODELS for testing. Provide isolation_model.pkl for production."
                )

            return True
        except Exception as e:
            logger.error(f"Critical error loading models: {e}")
            return False

    def _create_mock_isolation_forest(self) -> IsolationForest:
        """Create a simple mock Isolation Forest for testing."""
        try:
            # Generate synthetic training data
            np.random.seed(42)
            X_train = np.random.randn(1000, 5) * np.array([1, 0.5, 20, 0.5, 5]) + np.array(
                [12, 5, 100, 2, 40]
            )

            model = IsolationForest(
                n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1
            )
            model.fit(X_train)
            logger.info("Created mock Isolation Forest model")
            return model
        except Exception as e:
            logger.error(f"Failed to create mock Isolation Forest: {e}")
            return None

    def _create_mock_lof(self) -> LocalOutlierFactor:
        """Create a simple mock Local Outlier Factor model for testing."""
        try:
            # Generate synthetic training data
            np.random.seed(42)
            X_train = np.random.randn(1000, 5) * np.array([1, 0.5, 20, 0.5, 5]) + np.array(
                [12, 5, 100, 2, 40]
            )

            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
            model.fit(X_train)
            logger.info("Created mock Local Outlier Factor model")
            return model
        except Exception as e:
            logger.error(f"Failed to create mock LOF: {e}")
            return None

    def _create_mock_autoencoder(self) -> Any:
        """Create a simple mock Autoencoder model for testing."""
        try:
            if keras is None:
                logger.warning("TensorFlow/Keras not available. Cannot create mock Autoencoder.")
                return None

            # Create a simple autoencoder model
            input_dim = 5
            encoding_dim = 3

            # Encoder
            encoder_input = keras.layers.Input(shape=(input_dim,))
            encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoder_input)

            # Decoder
            decoded = keras.layers.Dense(input_dim, activation='linear')(encoded)

            # Autoencoder model
            autoencoder = keras.models.Model(encoder_input, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            # Train on synthetic data
            np.random.seed(42)
            X_train = np.random.randn(1000, input_dim) * np.array([1, 0.5, 20, 0.5, 5]) + np.array(
                [12, 5, 100, 2, 40]
            )

            # Normalize data
            X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

            autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)
            logger.info("Created mock Autoencoder model")
            return autoencoder
        except Exception as e:
            logger.error(f"Failed to create mock Autoencoder: {e}")
            return None

    def _load_lstm_model(self) -> bool:
        """Load LSTM model from models directory."""
        try:
            if keras is None:
                logger.warning("TensorFlow/Keras not available. LSTM model cannot be loaded.")
                self.model_sources["lstm"] = "unavailable"
                self.model_paths["lstm"] = None
                return False

            lstm_path = os.path.join(self.models_path, "lstm_model.keras")

            if not os.path.exists(lstm_path):
                logger.warning(f"LSTM model not found at {lstm_path}")
                self.model_sources["lstm"] = "unavailable"
                self.model_paths["lstm"] = None
                return False

            self.lstm_model = keras.models.load_model(lstm_path)
            self.model_sources["lstm"] = "custom"
            self.model_paths["lstm"] = lstm_path
            logger.info(f"✓ LSTM model loaded from {lstm_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LSTM model: {e}")
            self.model_sources["lstm"] = "unavailable"
            self.model_paths["lstm"] = None
            return False

    def _load_lof_model(self) -> bool:
        """Load Local Outlier Factor model from models directory."""
        try:
            lof_path = os.path.join(self.models_path, "lof_model.pkl")

            if not os.path.exists(lof_path):
                logger.warning(f"LOF model not found at {lof_path}. Creating mock model.")
                self.local_outlier_factor = self._create_mock_lof()
                self.model_sources["local_outlier_factor"] = "mock"
                self.model_paths["local_outlier_factor"] = None
                self.using_mock_models = True
                return True

            with open(lof_path, "rb") as f:
                self.local_outlier_factor = pickle.load(f)

            # predict/score_samples on unseen points require novelty=True.
            if not getattr(self.local_outlier_factor, "novelty", False):
                logger.warning(
                    "Loaded LOF model does not support novelty scoring. Replacing with mock novelty LOF."
                )
                self.local_outlier_factor = self._create_mock_lof()
                self.model_sources["local_outlier_factor"] = "mock"
                self.model_paths["local_outlier_factor"] = None
                self.using_mock_models = True
            else:
                self.model_sources["local_outlier_factor"] = "custom"
                self.model_paths["local_outlier_factor"] = lof_path

            logger.info(f"✓ LOF model loaded from {lof_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LOF model: {e}. Creating mock model.")
            self.local_outlier_factor = self._create_mock_lof()
            self.model_sources["local_outlier_factor"] = "mock"
            self.model_paths["local_outlier_factor"] = None
            self.using_mock_models = True
            return True

    def _load_autoencoder_model(self) -> bool:
        """Load Autoencoder model from models directory."""
        try:
            if keras is None:
                logger.warning("TensorFlow/Keras not available. Autoencoder model cannot be loaded.")
                self.model_sources["autoencoder"] = "unavailable"
                self.model_paths["autoencoder"] = None
                return False

            ae_path = os.path.join(self.models_path, "autoencoder_model.keras")

            if not os.path.exists(ae_path):
                logger.warning(f"Autoencoder model not found at {ae_path}. Creating mock model.")
                self.autoencoder_model = self._create_mock_autoencoder()
                self.model_sources["autoencoder"] = (
                    "mock" if self.autoencoder_model is not None else "unavailable"
                )
                self.model_paths["autoencoder"] = None
                self.using_mock_models = True
                return True

            self.autoencoder_model = keras.models.load_model(ae_path)
            self.model_sources["autoencoder"] = "custom"
            self.model_paths["autoencoder"] = ae_path
            logger.info(f"✓ Autoencoder model loaded from {ae_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Autoencoder model: {e}. Creating mock model.")
            self.autoencoder_model = self._create_mock_autoencoder()
            self.model_sources["autoencoder"] = (
                "mock" if self.autoencoder_model is not None else "unavailable"
            )
            self.model_paths["autoencoder"] = None
            self.using_mock_models = True
            return True

    def _load_prophet_models(self) -> bool:
        """Load all available Prophet models from models directory."""
        try:
            # Look for prophet_*.pkl files
            prophet_models_found = []

            if os.path.exists(self.models_path):
                for filename in os.listdir(self.models_path):
                    if filename.startswith("prophet_") and filename.endswith(".pkl"):
                        # Extract feature name from filename (e.g., prophet_battery_temp.pkl -> battery_temp)
                        feature_name = filename.replace("prophet_", "").replace(".pkl", "")
                        filepath = os.path.join(self.models_path, filename)

                        try:
                            with open(filepath, "rb") as f:
                                model = pickle.load(f)
                            self.prophet_models[feature_name] = model
                            prophet_models_found.append(feature_name)
                            logger.info(f"✓ Loaded Prophet model for '{feature_name}'")
                        except Exception as e:
                            logger.warning(f"Failed to load Prophet model '{filename}': {e}")

            if prophet_models_found:
                logger.info(f"Available Prophet models: {', '.join(prophet_models_found)}")
            else:
                logger.warning("No Prophet models found in models directory")

            return len(self.prophet_models) > 0
        except Exception as e:
            logger.warning(f"Error loading Prophet models: {e}")
            return False

    def get_isolation_forest(self) -> Optional[IsolationForest]:
        """Get Isolation Forest model."""
        return self.isolation_forest

    def get_lstm_model(self) -> Optional[Any]:
        """Get LSTM model."""
        return self.lstm_model

    def get_lof_model(self) -> Optional[LocalOutlierFactor]:
        """Get Local Outlier Factor model."""
        return self.local_outlier_factor

    def get_autoencoder_model(self) -> Optional[Any]:
        """Get Autoencoder model."""
        return self.autoencoder_model

    def get_prophet_model(self, feature_name: str) -> Optional[Any]:
        """Get Prophet model for a specific feature."""
        return self.prophet_models.get(feature_name)

    def get_prophet_models(self) -> Dict[str, Any]:
        """Get all available Prophet models."""
        return self.prophet_models

    def get_available_prophet_features(self) -> list:
        """Get list of available Prophet model features."""
        return list(self.prophet_models.keys())

    def get_model_source(self, model_name: str) -> str:
        """Get source for a model: custom | mock | unavailable."""
        return self.model_sources.get(model_name, "unavailable")

    def get_model_usage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return model usage metadata for API/debugging."""
        return {
            "isolation_forest": {
                "source": self.model_sources["isolation_forest"],
                "using_custom_model": self.model_sources["isolation_forest"] == "custom",
                "model_path": self.model_paths["isolation_forest"],
            },
            "local_outlier_factor": {
                "source": self.model_sources["local_outlier_factor"],
                "using_custom_model": self.model_sources["local_outlier_factor"] == "custom",
                "model_path": self.model_paths["local_outlier_factor"],
            },
            "lstm": {
                "source": self.model_sources["lstm"],
                "using_custom_model": self.model_sources["lstm"] == "custom",
                "model_path": self.model_paths["lstm"],
            },
            "autoencoder": {
                "source": self.model_sources["autoencoder"],
                "using_custom_model": self.model_sources["autoencoder"] == "custom",
                "model_path": self.model_paths["autoencoder"],
            },
        }

    def save_isolation_forest(self, path: Optional[str] = None) -> bool:
        """Save Isolation Forest model to disk."""
        try:
            if self.isolation_forest is None:
                logger.error("No Isolation Forest model to save")
                return False

            save_path = path or os.path.join(self.models_path, "isolation_forest.pkl")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(self.isolation_forest, f)
            logger.info(f"✓ Isolation Forest saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Isolation Forest: {e}")
            return False


# Global model manager instance
model_manager = ModelManager()
