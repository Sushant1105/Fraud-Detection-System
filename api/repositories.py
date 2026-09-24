"""
Data access repository for transaction predictions and analytics aggregations.
"""
import json
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.models import TransactionRecord
from api.schemas import TransactionInput, PredictionResponse


def save_transaction_prediction(
    db: Session,
    input_data: TransactionInput,
    prediction: PredictionResponse,
    algorithm: str = "XGBClassifier",
    version: str = "1.0.0",
) -> TransactionRecord:
    """
    Persists an evaluated transaction and its SHAP explanations into the database.
    Stores exact SHAP values without alteration.
    """
    # Serialize top SHAP contributing features directly from the prediction response
    serialized_features = json.dumps(
        [feat.model_dump() for feat in prediction.top_contributing_features]
    )

    record = TransactionRecord(
        transaction_id=prediction.transaction_id or input_data.transaction_id,
        time=input_data.Time,
        amount=input_data.Amount,
        fraud_probability=prediction.fraud_probability,
        threshold=prediction.threshold,
        is_fraud=prediction.is_fraud,
        decision=prediction.decision,
        risk_level=prediction.risk_level,
        model_algorithm=algorithm,
        model_version=version,
        latency_ms=prediction.latency_ms,
        top_contributing_features=serialized_features,
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_transactions(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[TransactionRecord], int]:
    """
    Retrieves a paginated list of transaction prediction records ordered newest first,
    along with the total count of transactions stored.
    """
    total = db.query(func.count(TransactionRecord.id)).scalar() or 0
    records = (
        db.query(TransactionRecord)
        .order_by(TransactionRecord.created_at.desc(), TransactionRecord.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return records, total


def get_transaction_by_id(
    db: Session,
    transaction_id: str,
) -> Optional[TransactionRecord]:
    """
    Finds a transaction record by the client-provided transaction_id.
    If multiple submissions share the same transaction_id, returns the latest record.
    """
    return (
        db.query(TransactionRecord)
        .filter(TransactionRecord.transaction_id == transaction_id)
        .order_by(TransactionRecord.created_at.desc(), TransactionRecord.id.desc())
        .first()
    )


def get_transaction_stats(db: Session) -> Dict[str, Any]:
    """
    Computes aggregate metrics from recorded transactions via SQL aggregates:
    - total_transactions
    - fraud_count
    - legitimate_count
    - fraud_rate (percentage)
    """
    total = db.query(func.count(TransactionRecord.id)).scalar() or 0
    fraud_count = (
        db.query(func.count(TransactionRecord.id))
        .filter(TransactionRecord.is_fraud.is_(True))
        .scalar()
        or 0
    )
    legitimate_count = total - fraud_count
    fraud_rate = round((fraud_count / total * 100.0), 2) if total > 0 else 0.0

    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "fraud_rate": fraud_rate,
    }
