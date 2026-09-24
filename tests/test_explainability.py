"""
Unit tests for FraudExplainer and explain_transaction in src/explainability.py.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.explainability import FraudExplainer, explain_transaction


@pytest.fixture(scope="module")
def explainer():
    return FraudExplainer()


@pytest.fixture(scope="module")
def sample_raw_tx():
    df = pd.read_csv("data/raw/creditcard.csv", nrows=5)
    df = df.drop(columns=["Class"])
    return df


@pytest.fixture(scope="module")
def sample_proc_tx():
    df = pd.read_csv("data/processed/test.csv", nrows=5)
    df = df.drop(columns=["Class"])
    return df


def test_explainer_initialization(explainer):
    assert explainer.model is not None
    assert explainer.preprocessor is not None
    assert explainer.threshold == 0.35
    assert len(explainer.feature_names) == 34
    assert np.isclose(explainer.base_value, -5.910916, atol=1e-3)
    assert np.isclose(explainer.base_probability, 0.002702, atol=1e-4)


def test_explain_instance_raw_dict(explainer, sample_raw_tx):
    raw_dict = sample_raw_tx.iloc[0].to_dict()
    res = explainer.explain_instance(raw_dict, top_k=5)

    assert "fraud_probability" in res
    assert "prediction" in res
    assert "decision" in res
    assert "risk_level" in res
    assert "base_value" in res
    assert "margin" in res
    assert "top_contributing_features" in res
    assert "shap_values" in res

    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["prediction"] in (0, 1)
    assert res["decision"] in ("FRAUD", "LEGITIMATE")
    assert res["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert len(res["top_contributing_features"]) == 5
    assert len(res["shap_values"]) == 34


def test_margin_additivity_and_sigmoid(explainer, sample_proc_tx):
    row = sample_proc_tx.iloc[0]
    res = explainer.explain_instance(row, top_k=5)

    # Check margin equals base_value + sum of SHAP values
    shap_sum = sum(res["shap_values"].values())
    expected_margin = res["base_value"] + shap_sum
    assert np.isclose(res["margin"], expected_margin, atol=1e-4)

    # Check sigmoid(margin) matches fraud_probability
    sig_prob = 1.0 / (1.0 + np.exp(-res["margin"]))
    assert np.isclose(res["fraud_probability"], sig_prob, atol=1e-5)


def test_relative_attribution_share(explainer, sample_proc_tx):
    row = sample_proc_tx.iloc[0]
    res = explainer.explain_instance(row, top_k=5)

    total_abs_shap = sum(abs(v) for v in res["shap_values"].values())
    for item in res["top_contributing_features"]:
        feat = item["feature"]
        abs_feat_shap = abs(res["shap_values"][feat])
        expected_share = round((abs_feat_shap / total_abs_shap) * 100.0, 2)
        assert np.isclose(item["relative_attribution_share"], expected_share, atol=0.05)


def test_risk_level_boundaries(explainer):
    assert explainer.get_risk_level(0.001) == "LOW"
    assert explainer.get_risk_level(0.099) == "LOW"
    assert explainer.get_risk_level(0.100) == "MEDIUM"
    assert explainer.get_risk_level(0.349) == "MEDIUM"
    assert explainer.get_risk_level(0.350) == "HIGH"
    assert explainer.get_risk_level(0.990) == "HIGH"


def test_explain_transaction_convenience(sample_raw_tx):
    raw_dict = sample_raw_tx.iloc[0].to_dict()
    res = explain_transaction(raw_dict, top_k=3)
    assert len(res["top_contributing_features"]) == 3
    assert res["decision"] == "LEGITIMATE"
