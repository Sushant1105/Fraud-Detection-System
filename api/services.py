"""
Service layer encapsulating fraud prediction and SHAP explainability.
"""
import time
from typing import Dict, Any, Optional
from pathlib import Path
from src.explainability import FraudExplainer
from api.schemas import (
    TransactionInput,
    PredictionResponse,
    FeatureContribution,
    HealthResponse,
)


class FraudService:
    """
    Singleton service managing the loaded XGBoost model, preprocessor,
    and TreeSHAP explainer for synchronous real-time inference.
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        threshold_config_path: Optional[Path] = None,
    ):
        # Initialize FraudExplainer which loads model, preprocessor, and threshold
        kwargs = {}
        if model_path:
            kwargs["model_path"] = model_path
        if preprocessor_path:
            kwargs["preprocessor_path"] = preprocessor_path
        if threshold_config_path:
            kwargs["threshold_config_path"] = threshold_config_path

        self.explainer = FraudExplainer(**kwargs)

    @property
    def threshold(self) -> float:
        return self.explainer.threshold

    def predict_and_explain(self, input_data: TransactionInput) -> PredictionResponse:
        """
        Score a single raw transaction, preprocess it, evaluate fraud probability,
        and generate top-5 SHAP feature contributions with relative attribution shares.
        """
        t0 = time.perf_counter()
        
        # Extract dictionary of features
        data_dict = input_data.model_dump()
        tx_id = data_dict.pop("transaction_id", None)

        # Execute inference and explainability pipeline
        raw_result = self.explainer.explain_instance(data_dict, top_k=5)
        
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # Convert top contributing features to Pydantic models
        top_features = [
            FeatureContribution(
                feature=item["feature"],
                feature_value=item["feature_value"],
                shap_value=item["shap_value"],
                direction=item["direction"],
                relative_attribution_share=item["relative_attribution_share"],
            )
            for item in raw_result["top_contributing_features"]
        ]

        return PredictionResponse(
            transaction_id=tx_id,
            fraud_probability=raw_result["fraud_probability"],
            is_fraud=bool(raw_result["prediction"] == 1),
            decision=raw_result["decision"],
            threshold=raw_result["threshold"],
            risk_level=raw_result["risk_level"],
            base_value=raw_result["base_value"],
            margin=raw_result["margin"],
            top_contributing_features=top_features,
            latency_ms=latency_ms,
        )

    def get_health_status(self) -> HealthResponse:
        """Return operational health status and model configuration."""
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_algorithm=type(self.explainer.model).__name__,
            decision_threshold=self.threshold,
            features_count=len(self.explainer.feature_names),
            version="1.0.0",
        )
