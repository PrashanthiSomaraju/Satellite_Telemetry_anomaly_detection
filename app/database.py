"""MongoDB database operations."""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from typing import List, Dict, Any
from datetime import datetime
import logging
from bson import ObjectId
from app.config import settings

logger = logging.getLogger(__name__)


class MongoDBManager:
    """MongoDB operations manager."""

    def __init__(self):
        self.mongodb_uri = settings.mongodb_uri
        self.db_name = settings.db_name
        self.client = None
        self.db = None

    def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]

            # Create collections if not exist
            self._create_collections()
            logger.info("✓ Connected to MongoDB successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to MongoDB: {e}")
            return False

    def _create_collections(self):
        """Create collections with indexes."""
        if "telemetry" not in self.db.list_collection_names():
            self.db.create_collection("telemetry")
            self.db.telemetry.create_index("timestamp")

        if "anomalies" not in self.db.list_collection_names():
            self.db.create_collection("anomalies")
            self.db.anomalies.create_index("timestamp")
            self.db.anomalies.create_index("is_anomaly")

        if "forecasts" not in self.db.list_collection_names():
            self.db.create_collection("forecasts")
            self.db.forecasts.create_index("timestamp")

    def _serialize_bson(self, value: Any) -> Any:
        """Recursively convert BSON types (ObjectId, nested docs/lists) to JSON-safe types."""
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, dict):
            return {k: self._serialize_bson(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_bson(v) for v in value]
        return value

    def insert_telemetry(self, data: Dict[str, Any]) -> str:
        """Insert telemetry record."""
        try:
            data["timestamp"] = datetime.utcnow()
            result = self.db.telemetry.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting telemetry: {e}")
            return None

    def insert_anomaly(self, data: Dict[str, Any]) -> str:
        """Insert anomaly record."""
        try:
            data["timestamp"] = datetime.utcnow()
            result = self.db.anomalies.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting anomaly: {e}")
            return None

    def insert_forecast(self, data: Dict[str, Any]) -> str:
        """Insert forecast record."""
        try:
            data["timestamp"] = datetime.utcnow()
            result = self.db.forecasts.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting forecast: {e}")
            return None

    def get_recent_telemetry(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent telemetry records."""
        try:
            records = list(self.db.telemetry.find().sort("timestamp", -1).limit(limit))
            return [self._serialize_bson(record) for record in records]
        except Exception as e:
            logger.error(f"Error fetching telemetry: {e}")
            return []

    def get_recent_anomalies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent anomaly records."""
        try:
            records = list(
                self.db.anomalies.find({"is_anomaly": True})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return [self._serialize_bson(record) for record in records]
        except Exception as e:
            logger.error(f"Error fetching anomalies: {e}")
            return []

    def get_telemetry_for_training(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get telemetry data for training."""
        try:
            records = list(
                self.db.telemetry.find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            records.reverse()
            return [self._serialize_bson(record) for record in records]
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return []

    def get_telemetry_for_retraining(self, window_size: int = 3000) -> List[Dict[str, Any]]:
        """Get telemetry data for retraining (sliding window)."""
        try:
            records = list(
                self.db.telemetry.find()
                .sort("timestamp", -1)
                .limit(window_size)
            )
            records.reverse()
            return [self._serialize_bson(record) for record in records]
        except Exception as e:
            logger.error(f"Error fetching data for retraining: {e}")
            return []

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Global instance
db_manager = MongoDBManager()
