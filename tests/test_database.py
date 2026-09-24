"""
Unit tests for the SQLite persistence layer, ORM models, and repository functions.
Follows strict test isolation using a temporary SQLite database.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, init_db
from api.models import TransactionRecord
from api.repositories import (
    save_transaction_prediction,
    get_transactions,
    get_transaction_by_id,
    get_transaction_stats,
)
from api.schemas import TransactionInput, PredictionResponse, FeatureContribution


@pytest.fixture
def test_db_session(tmp_path):
    """Provides an isolated SQLite database session for unit testing."""
    test_db_file = tmp_path / "test_fraud_detection.db"
    test_engine = create_engine(
        f"sqlite:///{test_db_file.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    init_db(engine_override=test_engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        test_engine.dispose()


def create_sample_input_and_prediction(tx_id="tx-100", is_fraud=False, prob=0.05, threshold=0.35, amount=100.0):
    """Helper to construct dummy inputs and predictions for database testing."""
    raw_input = TransactionInput(
        transaction_id=tx_id,
        Time=1234.0,
        Amount=amount,
        V1=0.1, V2=0.2, V3=0.3, V4=0.4, V5=0.5, V6=0.6, V7=0.7, V8=0.8,
        V9=0.9, V10=1.0, V11=1.1, V12=1.2, V13=1.3, V14=1.4, V15=1.5,
        V16=1.6, V17=1.7, V18=1.8, V19=1.9, V20=2.0, V21=2.1, V22=2.2,
        V23=2.3, V24=2.4, V25=2.5, V26=2.6, V27=2.7, V28=2.8,
    )
    prediction = PredictionResponse(
        transaction_id=tx_id,
        fraud_probability=prob,
        is_fraud=is_fraud,
        decision="FRAUD" if is_fraud else "LEGITIMATE",
        threshold=threshold,
        risk_level="HIGH" if is_fraud else "LOW",
        base_value=-4.5,
        margin=-2.0 if not is_fraud else 1.5,
        top_contributing_features=[
            FeatureContribution(
                feature="V14",
                feature_value=-1.25,
                shap_value=0.85 if is_fraud else -0.85,
                direction="increases_risk" if is_fraud else "decreases_risk",
                relative_attribution_share=45.5,
            )
        ],
        latency_ms=12.4,
    )
    return raw_input, prediction


def test_database_initialization(tmp_path):
    """Verify tables are created properly without error."""
    db_file = tmp_path / "init_check.db"
    eng = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine_override=eng)
    assert db_file.exists()
    eng.dispose()


def test_save_and_retrieve_transaction(test_db_session):
    """Verify saving a transaction prediction and reading it back."""
    inp, pred = create_sample_input_and_prediction(tx_id="tx-abc-123", is_fraud=False, prob=0.04)
    record = save_transaction_prediction(test_db_session, inp, pred)

    assert record.id is not None
    assert record.transaction_id == "tx-abc-123"
    assert record.decision == "LEGITIMATE"
    assert record.is_fraud is False
    assert record.amount == 100.0

    # Retrieve by transaction_id
    retrieved = get_transaction_by_id(test_db_session, "tx-abc-123")
    assert retrieved is not None
    assert retrieved.id == record.id
    record_dict = retrieved.to_dict()
    assert len(record_dict["top_contributing_features"]) == 1
    assert record_dict["top_contributing_features"][0]["feature"] == "V14"


def test_duplicate_transaction_id_returns_latest(test_db_session):
    """Verify that multiple records with the same transaction_id are stored and query returns latest."""
    inp1, pred1 = create_sample_input_and_prediction(tx_id="tx-duplicate", is_fraud=False, prob=0.10, amount=50.0)
    rec1 = save_transaction_prediction(test_db_session, inp1, pred1)

    inp2, pred2 = create_sample_input_and_prediction(tx_id="tx-duplicate", is_fraud=True, prob=0.85, amount=999.0)
    rec2 = save_transaction_prediction(test_db_session, inp2, pred2)

    # Both records should exist in the database
    records, total = get_transactions(test_db_session)
    assert total == 2
    assert rec1.id != rec2.id

    # get_transaction_by_id must return the latest record (rec2)
    latest = get_transaction_by_id(test_db_session, "tx-duplicate")
    assert latest is not None
    assert latest.id == rec2.id
    assert latest.decision == "FRAUD"
    assert latest.amount == 999.0


def test_nonexistent_transaction_returns_none(test_db_session):
    """Verify querying an unknown transaction_id returns None."""
    result = get_transaction_by_id(test_db_session, "tx-does-not-exist")
    assert result is None


def test_get_transactions_pagination(test_db_session):
    """Verify pagination and ordering (newest first)."""
    for i in range(5):
        inp, pred = create_sample_input_and_prediction(tx_id=f"tx-{i}", amount=float(i * 10))
        save_transaction_prediction(test_db_session, inp, pred)

    records, total = get_transactions(test_db_session, limit=2, offset=0)
    assert total == 5
    assert len(records) == 2
    # Newest record should be tx-4
    assert records[0].transaction_id == "tx-4"

    # Next page
    page2, _ = get_transactions(test_db_session, limit=2, offset=2)
    assert len(page2) == 2
    assert page2[0].transaction_id == "tx-2"


def test_get_transaction_stats(test_db_session):
    """Verify aggregate statistics calculation for empty and populated states."""
    # Empty DB
    empty_stats = get_transaction_stats(test_db_session)
    assert empty_stats["total_transactions"] == 0
    assert empty_stats["fraud_count"] == 0
    assert empty_stats["legitimate_count"] == 0
    assert empty_stats["fraud_rate"] == 0.0

    # Add 3 legitimate and 1 fraud
    for i in range(3):
        inp, pred = create_sample_input_and_prediction(tx_id=f"legit-{i}", is_fraud=False, prob=0.01)
        save_transaction_prediction(test_db_session, inp, pred)

    inp_f, pred_f = create_sample_input_and_prediction(tx_id="fraud-0", is_fraud=True, prob=0.92)
    save_transaction_prediction(test_db_session, inp_f, pred_f)

    stats = get_transaction_stats(test_db_session)
    assert stats["total_transactions"] == 4
    assert stats["fraud_count"] == 1
    assert stats["legitimate_count"] == 3
    assert stats["fraud_rate"] == 25.0
