"""
Integration and unit tests for FastAPI endpoints in api/main.py.
"""
import pytest
import json
from pathlib import Path
import pandas as pd
from starlette.testclient import TestClient

from api.main import app

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base, init_db, get_db

CONFIG_PATH = Path("models/xgboost_threshold_config.json")
RAW_DATA_PATH = Path("data/raw/creditcard.csv")


@pytest.fixture(scope="module")
def expected_threshold():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return float(cfg["selected_threshold"])


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Use isolated temporary SQLite database for automated test suite (Correction 8)
    temp_dir = tmp_path_factory.mktemp("api_test_db")
    test_db_file = temp_dir / "test_fraud_detection.db"
    test_engine = create_engine(
        f"sqlite:///{test_db_file.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    init_db(engine_override=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Use TestClient as context manager to invoke lifespan startup/shutdown
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    test_engine.dispose()



@pytest.fixture(scope="module")
def sample_legit_tx():
    df = pd.read_csv(RAW_DATA_PATH)
    legit_row = df[df["Class"] == 0].iloc[0].to_dict()
    del legit_row["Class"]
    legit_row["transaction_id"] = "test-legit-001"
    return legit_row


@pytest.fixture(scope="module")
def sample_fraud_tx():
    df = pd.read_csv(RAW_DATA_PATH)
    fraud_row = df[df["Class"] == 1].iloc[0].to_dict()
    del fraud_row["Class"]
    fraud_row["transaction_id"] = "test-fraud-001"
    return fraud_row


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["docs_url"]


def test_health_endpoint(client, expected_threshold):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_algorithm"] == "XGBClassifier"
    assert data["decision_threshold"] == expected_threshold
    assert data["features_count"] == 34


def test_predict_legitimate_transaction(client, sample_legit_tx, expected_threshold):
    response = client.post("/predict", json=sample_legit_tx)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "test-legit-001"
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["threshold"] == expected_threshold

    # Prediction relative to loaded threshold
    if data["fraud_probability"] >= expected_threshold:
        assert data["is_fraud"] is True
        assert data["decision"] == "FRAUD"
    else:
        assert data["is_fraud"] is False
        assert data["decision"] == "LEGITIMATE"

    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert len(data["top_contributing_features"]) == 5
    assert data["latency_ms"] is not None and data["latency_ms"] > 0

    # Validate relative attribution shares
    shares = [f["relative_attribution_share"] for f in data["top_contributing_features"]]
    assert all(0.0 <= s <= 100.0 for s in shares)


def test_predict_fraud_transaction(client, sample_fraud_tx, expected_threshold):
    response = client.post("/predict", json=sample_fraud_tx)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "test-fraud-001"
    assert "fraud_probability" in data
    assert data["threshold"] == expected_threshold

    # Verified relative to operating threshold
    if data["fraud_probability"] >= expected_threshold:
        assert data["is_fraud"] is True
        assert data["decision"] == "FRAUD"
        assert data["risk_level"] == "HIGH"
    else:
        assert data["is_fraud"] is False
        assert data["decision"] == "LEGITIMATE"


def test_predict_negative_amount_validation(client, sample_legit_tx):
    bad_payload = dict(sample_legit_tx)
    bad_payload["Amount"] = -25.50
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_missing_feature_validation(client, sample_legit_tx):
    bad_payload = dict(sample_legit_tx)
    del bad_payload["V14"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_forbidden_extra_field(client, sample_legit_tx):
    bad_payload = dict(sample_legit_tx)
    bad_payload["unexpected_client_field"] = "malicious_payload"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_margin_and_probability_consistency(client, sample_legit_tx):
    response = client.post("/predict", json=sample_legit_tx)
    assert response.status_code == 200
    data = response.json()

    import numpy as np
    margin = data["margin"]
    prob_calc = float(1.0 / (1.0 + np.exp(-margin)))
    assert np.isclose(data["fraud_probability"], prob_calc, atol=1e-4)


def test_get_transactions_endpoint(client):
    """Verify GET /transactions returns paginated list with valid structure."""
    response = client.get("/transactions?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()

    assert "transactions" in data
    assert "count" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert isinstance(data["transactions"], list)
    assert data["limit"] == 10
    assert data["count"] == len(data["transactions"])


def test_get_transaction_by_id_endpoint(client, sample_legit_tx):
    """Verify transaction saved during /predict can be retrieved via /transactions/{id}."""
    tx_payload = dict(sample_legit_tx)
    tx_payload["transaction_id"] = "test-history-inspect-01"

    predict_resp = client.post("/predict", json=tx_payload)
    assert predict_resp.status_code == 200
    pred_data = predict_resp.json()

    history_resp = client.get("/transactions/test-history-inspect-01")
    assert history_resp.status_code == 200
    hist_data = history_resp.json()

    assert hist_data["transaction_id"] == "test-history-inspect-01"
    assert hist_data["amount"] == tx_payload["Amount"]
    assert hist_data["decision"] == pred_data["decision"]
    assert hist_data["is_fraud"] == pred_data["is_fraud"]
    assert hist_data["fraud_probability"] == pred_data["fraud_probability"]
    assert len(hist_data["top_contributing_features"]) == len(pred_data["top_contributing_features"])
    assert hist_data["created_at"] is not None


def test_get_transaction_by_id_not_found(client):
    """Verify GET /transactions/{transaction_id} returns 404 for nonexistent id."""
    response = client.get("/transactions/unknown-nonexistent-tx-999")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_duplicate_transaction_id_api_returns_latest(client, sample_legit_tx):
    """Verify duplicate submissions return the latest record per Correction 4."""
    dup_id = "test-dup-client-tx"
    
    # First submission
    p1 = dict(sample_legit_tx)
    p1["transaction_id"] = dup_id
    p1["Amount"] = 45.00
    r1 = client.post("/predict", json=p1)
    assert r1.status_code == 200

    # Second submission with different amount
    p2 = dict(sample_legit_tx)
    p2["transaction_id"] = dup_id
    p2["Amount"] = 399.50
    r2 = client.post("/predict", json=p2)
    assert r2.status_code == 200

    # Lookup should return latest submission
    lookup = client.get(f"/transactions/{dup_id}")
    assert lookup.status_code == 200
    data = lookup.json()
    assert data["transaction_id"] == dup_id
    assert data["amount"] == 399.50


def test_stats_endpoint(client):
    """Verify /stats computes metrics dynamically without hard-coded assumptions (Correction 3)."""
    response = client.get("/stats")
    assert response.status_code == 200
    stats = response.json()

    assert "total_transactions" in stats
    assert "fraud_count" in stats
    assert "legitimate_count" in stats
    assert "fraud_rate" in stats

    assert stats["total_transactions"] >= 0
    assert stats["fraud_count"] >= 0
    assert stats["legitimate_count"] >= 0
    assert stats["total_transactions"] == stats["fraud_count"] + stats["legitimate_count"]

    if stats["total_transactions"] > 0:
        expected_rate = round(stats["fraud_count"] / stats["total_transactions"] * 100.0, 2)
        assert stats["fraud_rate"] == expected_rate
    else:
        assert stats["fraud_rate"] == 0.0

