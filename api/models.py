"""
SQLAlchemy ORM models for fraud detection persistence.
"""
import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from api.database import Base


class TransactionRecord(Base):
    """
    Persistent audit record of an evaluated financial transaction.
    Stores input identifiers, financial context, model scoring results,
    risk tiers, decision thresholds, and serialized SHAP feature contributions.
    """
    __tablename__ = "transaction_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    transaction_id = Column(String(100), nullable=True, index=True)
    time = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    decision = Column(String(20), nullable=False)
    risk_level = Column(String(20), nullable=False)
    model_algorithm = Column(String(50), nullable=False, default="XGBClassifier")
    model_version = Column(String(20), nullable=False, default="1.0.0")
    latency_ms = Column(Float, nullable=True)
    top_contributing_features = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    def to_dict(self):
        """Converts ORM model instance into JSON-compatible dictionary."""
        features = []
        if self.top_contributing_features:
            try:
                features = json.loads(self.top_contributing_features)
            except Exception:
                features = []

        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "time": self.time,
            "amount": self.amount,
            "fraud_probability": self.fraud_probability,
            "threshold": self.threshold,
            "is_fraud": self.is_fraud,
            "decision": self.decision,
            "risk_level": self.risk_level,
            "model_algorithm": self.model_algorithm,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "top_contributing_features": features,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
        }
