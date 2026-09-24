"""
Pydantic data schemas and contracts for the Real-Time Fraud Detection API.
"""
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class TransactionInput(BaseModel):
    """
    Raw transaction input schema.
    Clients must supply Time, Amount, and PCA components V1 through V28.
    Derived features (scaled_amount, log_amount, hour, hour_sin, hour_cos, scaled_time)
    are constructed internally via the production preprocessing pipeline.
    """
    transaction_id: Optional[str] = Field(
        default=None,
        description="Optional unique client transaction identifier for end-to-end tracing",
        examples=["tx-98234-eur"]
    )
    Time: float = Field(
        ...,
        description="Seconds elapsed between this transaction and the first transaction in the dataset",
        ge=0.0,
        examples=[406.0]
    )
    Amount: float = Field(
        ...,
        description="Transaction monetary amount in currency units (e.g. EUR)",
        ge=0.0,
        examples=[67.88]
    )
    V1: float = Field(..., description="PCA component 1", examples=[-2.3122265])
    V2: float = Field(..., description="PCA component 2", examples=[1.9519920])
    V3: float = Field(..., description="PCA component 3", examples=[-1.6098507])
    V4: float = Field(..., description="PCA component 4", examples=[3.9979056])
    V5: float = Field(..., description="PCA component 5", examples=[-0.5221879])
    V6: float = Field(..., description="PCA component 6", examples=[-1.4265453])
    V7: float = Field(..., description="PCA component 7", examples=[-2.5373873])
    V8: float = Field(..., description="PCA component 8", examples=[1.3916572])
    V9: float = Field(..., description="PCA component 9", examples=[-2.7700893])
    V10: float = Field(..., description="PCA component 10", examples=[-2.7722721])
    V11: float = Field(..., description="PCA component 11", examples=[3.2020332])
    V12: float = Field(..., description="PCA component 12", examples=[-2.8999074])
    V13: float = Field(..., description="PCA component 13", examples=[-0.5952219])
    V14: float = Field(..., description="PCA component 14", examples=[-4.2892538])
    V15: float = Field(..., description="PCA component 15", examples=[0.3897241])
    V16: float = Field(..., description="PCA component 16", examples=[-1.1407472])
    V17: float = Field(..., description="PCA component 17", examples=[-2.8300557])
    V18: float = Field(..., description="PCA component 18", examples=[-0.0168225])
    V19: float = Field(..., description="PCA component 19", examples=[0.4169557])
    V20: float = Field(..., description="PCA component 20", examples=[0.1269106])
    V21: float = Field(..., description="PCA component 21", examples=[0.5172324])
    V22: float = Field(..., description="PCA component 22", examples=[-0.0350494])
    V23: float = Field(..., description="PCA component 23", examples=[-0.4652111])
    V24: float = Field(..., description="PCA component 24", examples=[0.3201982])
    V25: float = Field(..., description="PCA component 25", examples=[0.0445192])
    V26: float = Field(..., description="PCA component 26", examples=[0.1778398])
    V27: float = Field(..., description="PCA component 27", examples=[0.2611450])
    V28: float = Field(..., description="PCA component 28", examples=[-0.1432759])

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_id": "tx-406-sample",
                "Time": 406.0,
                "Amount": 67.88,
                "V1": -2.3122265, "V2": 1.9519920, "V3": -1.6098507, "V4": 3.9979056,
                "V5": -0.5221879, "V6": -1.4265453, "V7": -2.5373873, "V8": 1.3916572,
                "V9": -2.7700893, "V10": -2.7722721, "V11": 3.2020332, "V12": -2.8999074,
                "V13": -0.5952219, "V14": -4.2892538, "V15": 0.3897241, "V16": -1.1407472,
                "V17": -2.8300557, "V18": -0.0168225, "V19": 0.4169557, "V20": 0.1269106,
                "V21": 0.5172324, "V22": -0.0350494, "V23": -0.4652111, "V24": 0.3201982,
                "V25": 0.0445192, "V26": 0.1778398, "V27": 0.2611450, "V28": -0.1432759
            }
        }
    )


class FeatureContribution(BaseModel):
    """Local feature attribution calculated via SHAP."""
    feature: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Raw value of the feature for this transaction")
    shap_value: float = Field(..., description="Shapley attribution value in log-odds/margin space")
    direction: str = Field(..., description="'increases_risk' (positive SHAP) or 'decreases_risk' (negative SHAP)")
    relative_attribution_share: float = Field(..., description="Percentage of total absolute attribution (|phi_i| / sum(|phi|)) * 100")


class PredictionResponse(BaseModel):
    """Complete prediction and explainability response schema."""
    transaction_id: Optional[str] = Field(default=None, description="Client transaction identifier")
    fraud_probability: float = Field(..., description="Calibrated fraud probability in [0.0, 1.0]", ge=0.0, le=1.0)
    is_fraud: bool = Field(..., description="True if fraud_probability >= threshold, otherwise False")
    decision: str = Field(..., description="'FRAUD' or 'LEGITIMATE'")
    threshold: float = Field(..., description="Operating decision threshold loaded from configuration")
    risk_level: str = Field(..., description="Risk tier: 'LOW', 'MEDIUM', or 'HIGH'")
    base_value: float = Field(..., description="Explainer expected base value in log-odds margin space")
    margin: float = Field(..., description="Final model log-odds margin output (base_value + sum(shap_values))")
    top_contributing_features: List[FeatureContribution] = Field(..., description="Top K features driving this decision")
    latency_ms: Optional[float] = Field(default=None, description="Inference and explanation elapsed latency in milliseconds")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="UTC timestamp of the prediction")


class HealthResponse(BaseModel):
    """API health status and model configuration schema."""
    status: str = Field(default="healthy", description="Operational health status")
    model_loaded: bool = Field(..., description="True if the ML model is loaded and ready for scoring")
    model_algorithm: str = Field(..., description="Loaded model algorithm name")
    decision_threshold: float = Field(..., description="Dynamically loaded operational decision threshold")
    features_count: int = Field(..., description="Number of expected features in the model pipeline")
    version: str = Field(default="1.0.0", description="API version")


class TransactionHistoryItem(BaseModel):
    """Persisted transaction prediction audit record schema."""
    id: int = Field(..., description="Unique database record identifier")
    transaction_id: Optional[str] = Field(default=None, description="Client-provided transaction identifier")
    time: float = Field(..., description="Transaction elapsed time feature")
    amount: float = Field(..., description="Monetary transaction amount")
    fraud_probability: float = Field(..., description="Model calculated fraud probability", ge=0.0, le=1.0)
    threshold: float = Field(..., description="Operating threshold used for evaluation")
    is_fraud: bool = Field(..., description="True if fraud, False if legitimate")
    decision: str = Field(..., description="Classification outcome: 'FRAUD' or 'LEGITIMATE'")
    risk_level: str = Field(..., description="Assigned risk tier: 'LOW', 'MEDIUM', or 'HIGH'")
    model_algorithm: str = Field(default="XGBClassifier", description="Model algorithm used")
    model_version: str = Field(default="1.0.0", description="Model version")
    latency_ms: Optional[float] = Field(default=None, description="Scoring latency in milliseconds")
    top_contributing_features: List[FeatureContribution] = Field(
        default_factory=list, description="Top SHAP feature attributions"
    )
    created_at: str = Field(..., description="UTC timestamp of the persistent record")


class TransactionListResponse(BaseModel):
    """Paginated list of historical transaction predictions."""
    transactions: List[TransactionHistoryItem] = Field(..., description="List of recorded transactions")
    count: int = Field(..., description="Number of transactions returned in this response")
    total: int = Field(..., description="Total count of transactions matching filter in database")
    limit: int = Field(..., description="Query limit applied")
    offset: int = Field(..., description="Query offset applied")


class TransactionStatsResponse(BaseModel):
    """Aggregated transaction scoring metrics across stored database history."""
    total_transactions: int = Field(..., description="Total transactions evaluated and stored in database")
    fraud_count: int = Field(..., description="Total transactions flagged as FRAUD")
    legitimate_count: int = Field(..., description="Total transactions classified as LEGITIMATE")
    fraud_rate: float = Field(..., description="Percentage of evaluated transactions classified as fraud")

