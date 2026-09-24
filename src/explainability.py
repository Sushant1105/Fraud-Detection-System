"""
Model explainability and interpretability module for the Real-Time Fraud Detection System.
Implements local and global transaction risk attribution using SHAP TreeExplainer.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "xgboost_advanced_model.joblib"
DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
DEFAULT_CONFIG_PATH = MODELS_DIR / "xgboost_threshold_config.json"


class FraudExplainer:
    """
    Production-grade model explainer using SHAP TreeExplainer for XGBoost.
    
    Provides mathematically exact Shapley value risk attribution in margin (log-odds)
    space, converted to fraud probabilities via the logistic sigmoid function:
        margin(x) = base_value + sum(shap_values)
        P(fraud) = sigmoid(margin(x)) = 1 / (1 + exp(-margin(x)))
        
    Relative attribution share is computed from absolute SHAP values:
        relative_attribution_share_i = (|phi_i| / sum(|phi_j|)) * 100%
    """
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
        threshold_config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    ):
        model_path = Path(model_path)
        preprocessor_path = Path(preprocessor_path)
        threshold_config_path = Path(threshold_config_path)

        # Support project-relative resolution if relative path doesn't exist in current working directory
        if not model_path.is_absolute() and not model_path.exists():
            candidate = PROJECT_ROOT / model_path
            if candidate.exists():
                model_path = candidate

        if not preprocessor_path.is_absolute() and not preprocessor_path.exists():
            candidate = PROJECT_ROOT / preprocessor_path
            if candidate.exists():
                preprocessor_path = candidate

        if not threshold_config_path.is_absolute() and not threshold_config_path.exists():
            candidate = PROJECT_ROOT / threshold_config_path
            if candidate.exists():
                threshold_config_path = candidate

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {preprocessor_path}")
        if not threshold_config_path.exists():
            raise FileNotFoundError(f"Threshold config not found at: {threshold_config_path}")


        # 1. Load trained model, preprocessor, and threshold metadata
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        with open(threshold_config_path, "r", encoding="utf-8") as f:
            self.threshold_config = json.load(f)

        # 2. Extract calibrated threshold and feature schema
        self.threshold = float(self.threshold_config.get("selected_threshold", 0.35))
        
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            self.feature_names = [str(col) for col in self.model.feature_names_in_]
        elif hasattr(self.preprocessor, "feature_names") and self.preprocessor.feature_names is not None:
            self.feature_names = [str(col) for col in self.preprocessor.feature_names]
        else:
            pca_cols = [f"V{i}" for i in range(1, 29)]
            eng_cols = ["scaled_amount", "log_amount", "scaled_time", "hour", "hour_sin", "hour_cos"]
            self.feature_names = pca_cols + eng_cols

        # 3. Reconstruct TreeExplainer from the saved model (zero serialization coupling)
        self.explainer = shap.TreeExplainer(self.model)
        # Prime explainer with dummy input to calculate exact base_value
        dummy_sv = self.explainer(np.zeros((1, len(self.feature_names))))
        self.base_value = float(dummy_sv.base_values[0])
        self.base_probability = float(1.0 / (1.0 + np.exp(-self.base_value)))

    def _prepare_input(self, transaction: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and align transaction input, applying preprocessing if raw features are provided.
        """
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.Series):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.DataFrame):
            df = transaction.copy()
        else:
            raise TypeError("Expected input as dict, pandas Series, or pandas DataFrame.")

        # Check if input already has all preprocessed features
        has_preprocessed = all(col in df.columns for col in self.feature_names)
        if has_preprocessed:
            return df[self.feature_names].copy()

        # Check if input has raw transaction features
        raw_required = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        has_raw = all(col in df.columns for col in raw_required)
        if has_raw:
            transformed = self.preprocessor.transform(df[raw_required])
            return transformed[self.feature_names].copy()

        missing_raw = [c for c in raw_required if c not in df.columns]
        missing_proc = [c for c in self.feature_names if c not in df.columns]
        raise ValueError(
            f"Input features could not be matched. "
            f"For raw input, missing: {missing_raw[:5]}... "
            f"For preprocessed input, missing: {missing_proc[:5]}..."
        )

    def get_risk_level(self, probability: float) -> str:
        """
        Categorize transaction into operational risk tiers.
        - HIGH: >= calibrated threshold (e.g. >= 0.35, candidate for blocking/manual review)
        - MEDIUM: >= 0.10 and < calibrated threshold (elevated risk, candidate for 2FA challenge)
        - LOW: < 0.10 (normal automated clearance)
        """
        if probability >= self.threshold:
            return "HIGH"
        elif probability >= 0.10:
            return "MEDIUM"
        else:
            return "LOW"

    def explain_instance(
        self,
        transaction: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate exact SHAP risk attribution for a single transaction.
        
        Args:
            transaction: Transaction features (raw or preprocessed).
            top_k: Number of highest-attribution features to highlight.
            
        Returns:
            Dictionary containing probability, prediction, risk tier, margin,
            top contributing features with relative attribution share, and full SHAP dict.
        """
        X = self._prepare_input(transaction)
        if len(X) != 1:
            raise ValueError("explain_instance expects a single transaction row.")

        # Compute exact Shapley values in margin space
        shap_explanation = self.explainer(X)
        shap_values_array = shap_explanation.values[0]
        base_value = float(shap_explanation.base_values[0])

        # Margin additivity: margin = base_value + sum(shap_values)
        margin = float(base_value + np.sum(shap_values_array))
        fraud_prob = float(1.0 / (1.0 + np.exp(-margin)))
        prediction = int(fraud_prob >= self.threshold)
        risk_level = self.get_risk_level(fraud_prob)

        # Attribution shares based on absolute SHAP values
        abs_shap = np.abs(shap_values_array)
        total_abs_shap = float(np.sum(abs_shap))

        # Rank features by magnitude of attribution
        sorted_indices = np.argsort(abs_shap)[::-1]
        top_features: List[Dict[str, Any]] = []

        for idx in sorted_indices[:top_k]:
            feat = str(self.feature_names[idx])
            val = float(shap_values_array[idx])
            raw_feat_val = float(X[feat].iloc[0])
            share = float((abs(val) / total_abs_shap) * 100.0) if total_abs_shap > 0 else 0.0

            top_features.append({
                "feature": feat,
                "feature_value": round(raw_feat_val, 6),
                "shap_value": round(val, 6),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "relative_attribution_share": round(share, 2),
            })

        shap_dict = {
            str(feat): round(float(val), 6)
            for feat, val in zip(self.feature_names, shap_values_array)
        }

        return {
            "fraud_probability": round(fraud_prob, 6),
            "prediction": prediction,
            "decision": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "threshold": self.threshold,
            "risk_level": risk_level,
            "base_value": round(base_value, 6),
            "base_probability": round(float(1.0 / (1.0 + np.exp(-base_value))), 6),
            "margin": round(margin, 6),
            "top_contributing_features": top_features,
            "shap_values": shap_dict,
        }

    def explain_batch(
        self,
        transactions: pd.DataFrame,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of transactions.
        """
        X = self._prepare_input(transactions)
        shap_explanation = self.explainer(X)
        shap_values_matrix = shap_explanation.values
        base_values = shap_explanation.base_values

        results = []
        for i in range(len(X)):
            sv = shap_values_matrix[i]
            bv = float(base_values[i])
            margin = float(bv + np.sum(sv))
            prob = float(1.0 / (1.0 + np.exp(-margin)))
            pred = int(prob >= self.threshold)
            risk = self.get_risk_level(prob)

            abs_sv = np.abs(sv)
            total_abs = float(np.sum(abs_sv))
            sorted_indices = np.argsort(abs_sv)[::-1]

            top_feats = []
            for idx in sorted_indices[:top_k]:
                feat = str(self.feature_names[idx])
                val = float(sv[idx])
                raw_v = float(X[feat].iloc[i])
                share = float((abs(val) / total_abs) * 100.0) if total_abs > 0 else 0.0
                top_feats.append({
                    "feature": feat,
                    "feature_value": round(raw_v, 6),
                    "shap_value": round(val, 6),
                    "direction": "increases_risk" if val > 0 else "decreases_risk",
                    "relative_attribution_share": round(share, 2),
                })

            results.append({
                "fraud_probability": round(prob, 6),
                "prediction": pred,
                "decision": "FRAUD" if pred == 1 else "LEGITIMATE",
                "threshold": self.threshold,
                "risk_level": risk,
                "base_value": round(bv, 6),
                "base_probability": round(float(1.0 / (1.0 + np.exp(-bv))), 6),
                "margin": round(margin, 6),
                "top_contributing_features": top_feats,
            })

        return results


def explain_transaction(
    transaction: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    explainer: Optional[FraudExplainer] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function to explain an incoming transaction.
    If no explainer instance is passed, initializes a default FraudExplainer.
    """
    if explainer is None:
        explainer = FraudExplainer()
    return explainer.explain_instance(transaction, top_k=top_k)
