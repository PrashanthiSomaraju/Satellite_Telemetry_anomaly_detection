"""Anomaly detection using Isolation Forest, Local Outlier Factor, LSTM, and Autoencoder."""

import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Multi-model ensemble anomaly detection using Isolation Forest, LOF, LSTM, and Autoencoder."""

    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
        self.feature_names = [
            "battery_voltage",
            "bus_voltage",
            "wheel_rpm",
            "current",
            "wheel_temperature"
        ]
        self.sequence_length = 10  # For LSTM: look at 10 previous timesteps
        # Conservative safety bounds; values outside these ranges are always anomalous.
        self.hard_bounds = {
            "battery_voltage": (8.0, 16.8),
            "bus_voltage": (3.0, 12.0),
            "wheel_rpm": (0.0, 5000.0),
            "current": (0.0, 10.0),
            "wheel_temperature": (-40.0, 120.0),
        }

    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from telemetry data."""
        features = np.array(
            [data.get(f, 0.0) for f in self.feature_names],
            dtype=np.float32
        )
        return features.reshape(1, -1)

    def detect(self, data: Dict[str, Any]) -> Tuple[bool, float, Dict]:
        """
        Detect anomaly using ensemble of multiple models:
        - Isolation Forest
        - Local Outlier Factor (LOF)
        - LSTM
        - Autoencoder

        Returns:
            Tuple of (is_anomaly, confidence, details_dict)
        """
        try:
            results = {
                "isolation_forest": {"available": False, "is_anomaly": False, "confidence": 0.0},
                "local_outlier_factor": {"available": False, "is_anomaly": False, "confidence": 0.0},
                "lstm": {"available": False, "is_anomaly": False, "confidence": 0.0},
                "autoencoder": {"available": False, "is_anomaly": False, "confidence": 0.0},
                "hard_limit": {"available": True, "is_anomaly": False, "confidence": 0.0, "violations": []},
                "ensemble_anomaly": False,
                "ensemble_confidence": 0.0,
                "detection_method": "ensemble (IF + LOF + LSTM + Autoencoder)"
            }

            # Hard-limit detection first for obvious physical outliers.
            hard_limit_score = self._detect_hard_limits(data)
            results["hard_limit"] = hard_limit_score

            # Isolation Forest detection
            if_score = self._detect_isolation_forest(data)
            if if_score:
                results["isolation_forest"] = if_score

            # Local Outlier Factor detection
            lof_score = self._detect_local_outlier_factor(data)
            if lof_score:
                results["local_outlier_factor"] = lof_score

            # LSTM detection
            lstm_score = self._detect_lstm(data)
            if lstm_score:
                results["lstm"] = lstm_score

            # Autoencoder detection
            ae_score = self._detect_autoencoder(data)
            if ae_score:
                results["autoencoder"] = ae_score

            # Annotate model results with whether they are custom/mock/unavailable.
            source_map = {
                "isolation_forest": "isolation_forest",
                "local_outlier_factor": "local_outlier_factor",
                "lstm": "lstm",
                "autoencoder": "autoencoder",
            }
            for result_key, model_key in source_map.items():
                source = self.model_manager.get_model_source(model_key)
                results[result_key]["source"] = source
                results[result_key]["using_custom_model"] = source == "custom"

            model_keys = ["isolation_forest", "local_outlier_factor", "lstm", "autoencoder"]
            active_models = [k for k in model_keys if results[k].get("available")]
            anomaly_votes = sum(1 for k in active_models if results[k]["is_anomaly"])

            # Strict majority across active models; if only one model is active, trust that model.
            if len(active_models) <= 1:
                required_votes = 1
            else:
                required_votes = (len(active_models) // 2) + 1
            model_ensemble_anomaly = anomaly_votes >= required_votes

            # Hard limits override model votes for obvious safety violations.
            results["ensemble_anomaly"] = bool(results["hard_limit"]["is_anomaly"] or model_ensemble_anomaly)

            confidences = [results[k]["confidence"] for k in active_models]
            if confidences:
                model_confidence = float(np.mean(confidences))
            else:
                model_confidence = 0.0

            results["ensemble_confidence"] = max(model_confidence, results["hard_limit"]["confidence"])

            return results["ensemble_anomaly"], results["ensemble_confidence"], results

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0, {"error": str(e)}

    def _detect_hard_limits(self, data: Dict[str, Any]) -> Dict:
        """Flag anomalies when values violate conservative physical bounds."""
        violations = []
        for feature, (lower, upper) in self.hard_bounds.items():
            value = data.get(feature)
            if value is None:
                continue
            if value < lower or value > upper:
                violations.append(
                    {
                        "feature": feature,
                        "value": float(value),
                        "expected_min": lower,
                        "expected_max": upper,
                    }
                )

        is_anomaly = len(violations) > 0
        confidence = min(1.0, 0.7 + 0.1 * len(violations)) if is_anomaly else 0.0
        return {
            "available": True,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "violations": violations,
        }

    def _detect_isolation_forest(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using Isolation Forest."""
        try:
            if_model = self.model_manager.get_isolation_forest()
            if if_model is None:
                return {"available": False, "is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)

            # IF returns -1 for anomalies, 1 for normal
            predictions = if_model.predict(features)
            decision = if_model.decision_function(features)

            is_anomaly = predictions[0] == -1
            # Lower decision_function means more anomalous.
            confidence = 1 / (1 + np.exp(6.0 * decision[0]))

            return {
                "available": True,
                "is_anomaly": bool(is_anomaly),
                "confidence": float(np.clip(confidence, 0.0, 1.0))
            }
        except Exception as e:
            logger.warning(f"Isolation Forest detection failed: {e}")
            return {"available": False, "is_anomaly": False, "confidence": 0.0}

    def _detect_lstm(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using LSTM model."""
        try:
            lstm_model = self.model_manager.get_lstm_model()
            if lstm_model is None:
                return {"available": False, "is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)

            # LSTM expects 3D input: (batch_size, sequence_length, features)
            # Create a simple sequence by repeating the current sample
            sequence = np.repeat(features, self.sequence_length, axis=0).reshape(1, self.sequence_length, -1)

            # Get prediction (assumes model outputs anomaly score between 0 and 1)
            prediction = lstm_model.predict(sequence, verbose=0)
            confidence = float(prediction[0][0]) if prediction.shape[-1] > 0 else 0.0

            # Threshold for anomaly
            threshold = 0.5
            is_anomaly = confidence > threshold

            return {
                "available": True,
                "is_anomaly": bool(is_anomaly),
                "confidence": min(confidence, 1.0)  # Ensure it's between 0 and 1
            }
        except Exception as e:
            logger.warning(f"LSTM detection failed: {e}")
            return {"available": False, "is_anomaly": False, "confidence": 0.0}

    def _detect_local_outlier_factor(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using Local Outlier Factor (LOF)."""
        try:
            lof_model = self.model_manager.get_lof_model()
            if lof_model is None:
                return {"available": False, "is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)

            # LOF returns -1 for outliers, 1 for inliers
            predictions = lof_model.predict(features)
            is_anomaly = predictions[0] == -1

            # score_samples is higher for inliers and lower for outliers.
            sample_score = float(lof_model.score_samples(features)[0])
            offset = float(getattr(lof_model, "offset_", -1.5))
            confidence = 1 / (1 + np.exp(-6.0 * (offset - sample_score)))

            return {
                "available": True,
                "is_anomaly": bool(is_anomaly),
                "confidence": float(np.clip(confidence, 0, 1))
            }
        except Exception as e:
            logger.warning(f"Local Outlier Factor detection failed: {e}")
            return {"available": False, "is_anomaly": False, "confidence": 0.0}

    def _detect_autoencoder(self, data: Dict[str, Any]) -> Dict:
        """Detect anomaly using Autoencoder reconstruction error."""
        try:
            autoencoder_model = self.model_manager.get_autoencoder_model()
            if autoencoder_model is None:
                return {"available": False, "is_anomaly": False, "confidence": 0.0}

            features = self.prepare_features(data)

            # Autoencoder: compute reconstruction error
            reconstruction = autoencoder_model.predict(features, verbose=0)

            # Calculate Mean Squared Error (MSE) as anomaly score
            mse = np.mean((features - reconstruction) ** 2)

            # Use a threshold or normalized score
            # Normalize using sigmoid for confidence
            threshold = 0.05  # Typical threshold for MSE
            confidence = min(mse / threshold, 1.0)  # Confidence increases with error
            is_anomaly = mse > threshold

            return {
                "available": True,
                "is_anomaly": bool(is_anomaly),
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.warning(f"Autoencoder detection failed: {e}")
            return {"available": False, "is_anomaly": False, "confidence": 0.0}
