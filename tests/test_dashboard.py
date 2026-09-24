"""
Unit and integration tests for dashboard components and API client.
"""
import pytest
import threading
import time
import uvicorn
from dashboard.services.api_client import FraudApiClient
from dashboard.utils.presets import LEGITIMATE_SAMPLE, FRAUD_SAMPLE
from dashboard.components.telemetry import record_transaction, init_telemetry_state


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import init_db, get_db


class LiveServer:
    def __init__(self, app, port=8011):
        self.port = port
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        time.sleep(1.2)

    def stop(self):
        self.server.should_exit = True
        time.sleep(0.3)


@pytest.fixture(scope="module")
def running_server(tmp_path_factory):
    # Use isolated temporary SQLite database for automated test suite (Correction 8)
    temp_dir = tmp_path_factory.mktemp("dash_test_db")
    test_db_file = temp_dir / "test_dash_fraud.db"
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

    from api.main import app
    app.dependency_overrides[get_db] = override_get_db

    srv = LiveServer(app, port=8011)
    srv.start()
    yield "http://127.0.0.1:8011"
    srv.stop()
    app.dependency_overrides.clear()
    test_engine.dispose()


def test_api_client_offline_graceful():
    """Verify that client returns graceful error rather than crashing when API is unreachable."""
    offline_client = FraudApiClient(base_url="http://127.0.0.1:59999")
    is_online, data, err = offline_client.check_health(timeout=1.0)
    assert is_online is False
    assert data is None
    assert "Cannot connect" in err or "Connection Error" in err

    success, pred, p_err = offline_client.predict_transaction(LEGITIMATE_SAMPLE, timeout=1.0)
    assert success is False
    assert pred is None
    assert "Connection Error" in p_err or "Unable to reach" in p_err

    # Check offline handling for history and stats methods
    h_succ, h_data, h_err = offline_client.get_transactions(timeout=1.0)
    assert h_succ is False
    assert h_data is None
    assert "Connection Error" in h_err or "Unable to reach" in h_err

    s_succ, s_data, s_err = offline_client.get_stats(timeout=1.0)
    assert s_succ is False
    assert s_data is None
    assert "Connection Error" in s_err or "Unable to reach" in s_err



def test_api_client_live_health(running_server):
    client = FraudApiClient(base_url=running_server)
    is_online, data, err = client.check_health()
    assert is_online is True
    assert err is None
    assert data["status"] == "healthy"
    assert "decision_threshold" in data


def test_api_client_live_legit_prediction(running_server):
    client = FraudApiClient(base_url=running_server)
    success, data, err = client.predict_transaction(LEGITIMATE_SAMPLE)
    assert success is True
    assert err is None

    threshold = data["threshold"]
    prob = data["fraud_probability"]

    # Verify decision relative to returned threshold
    if prob >= threshold:
        assert data["is_fraud"] is True
        assert data["decision"] == "FRAUD"
    else:
        assert data["is_fraud"] is False
        assert data["decision"] == "LEGITIMATE"

    assert len(data["top_contributing_features"]) == 5
    # Verify that features have required fields without hardcoding specific names
    for item in data["top_contributing_features"]:
        assert "feature" in item
        assert "shap_value" in item
        assert item["direction"] in ("increases_risk", "decreases_risk")
        assert 0.0 <= item["relative_attribution_share"] <= 100.0


def test_api_client_live_fraud_prediction(running_server):
    client = FraudApiClient(base_url=running_server)
    success, data, err = client.predict_transaction(FRAUD_SAMPLE)
    assert success is True
    assert err is None

    threshold = data["threshold"]
    prob = data["fraud_probability"]

    # Verify decision relative to returned threshold
    if prob >= threshold:
        assert data["is_fraud"] is True
        assert data["decision"] == "FRAUD"
        assert data["risk_level"] == "HIGH"
    else:
        assert data["is_fraud"] is False
        assert data["decision"] == "LEGITIMATE"


def test_presets_purity():
    """Ensure presets contain ONLY raw features and no derived features."""
    forbidden = ["scaled_amount", "log_amount", "scaled_time", "hour", "hour_sin", "hour_cos"]
    for sample in (LEGITIMATE_SAMPLE, FRAUD_SAMPLE):
        for f in forbidden:
            assert f not in sample
        assert "Time" in sample
        assert "Amount" in sample
        assert all(f"V{i}" in sample for i in range(1, 29))


def test_api_client_history_and_stats(running_server):
    """Verify FraudApiClient history and stats retrieval from running server."""
    client = FraudApiClient(base_url=running_server)

    # Initially or after previous tests, query stats
    stats_ok, stats_data, s_err = client.get_stats()
    assert stats_ok is True
    assert s_err is None
    assert "total_transactions" in stats_data
    assert "fraud_count" in stats_data

    # Submit a transaction
    pred_ok, pred_data, p_err = client.predict_transaction(LEGITIMATE_SAMPLE)
    assert pred_ok is True
    assert pred_data is not None

    # Fetch transaction list
    hist_ok, hist_data, h_err = client.get_transactions(limit=10)
    assert hist_ok is True
    assert h_err is None
    assert "transactions" in hist_data
    assert hist_data["count"] >= 1

    # Fetch single transaction using client transaction_id
    tx_id = LEGITIMATE_SAMPLE.get("transaction_id")
    if tx_id:
        single_ok, single_data, single_err = client.get_transaction(tx_id)
        assert single_ok is True
        assert single_err is None
        assert single_data["transaction_id"] == tx_id
        assert "top_contributing_features" in single_data

