"""
FastAPI application exposing real-time fraud scoring and SHAP explainability.
"""
import logging
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.schemas import (
    TransactionInput,
    PredictionResponse,
    HealthResponse,
    TransactionHistoryItem,
    TransactionListResponse,
    TransactionStatsResponse,
)
from api.services import FraudService
from api.database import init_db, get_db
from api.repositories import (
    save_transaction_prediction,
    get_transactions,
    get_transaction_by_id,
    get_transaction_stats,
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes database tables, loads ML models, preprocessor, and TreeSHAP explainer once at startup,
    storing the singleton service in app.state.
    """
    print("[Startup] Initializing database tables...")
    init_db()
    print("[Startup] Initializing FraudService and loading pipeline artifacts...")
    app.state.fraud_service = FraudService()
    print(f"[Startup] FraudService ready. Decision threshold: tau = {app.state.fraud_service.threshold}")
    yield
    print("[Shutdown] Cleaning up API resources.")



app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="""
## Production Real-Time Fraud Detection & SHAP Explainability Service

This RESTful microservice evaluates incoming financial transactions in real time:
- **Feature Pipeline**: Automatic preprocessing and circadian/amount feature engineering.
- **Scoring Engine**: Evaluates transactions with the calibrated Phase 7 XGBoost classifier.
- **Decision Engine**: Classifies fraud using the dynamically loaded operating threshold.
- **Explainability**: Generates exact TreeSHAP margin-space attributions, highlighting the top 5 risk drivers with relative attribution shares.

### Endpoints:
- `GET /health`: Model status, configuration, and decision threshold.
- `POST /predict`: Real-time transaction scoring and risk attribution.
- `GET /docs`: Interactive Swagger UI documentation.
- `GET /redoc`: ReDoc technical documentation.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Optional CORS middleware for web dashboard connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service(request: Request) -> FraudService:
    """Dependency helper to extract the loaded FraudService singleton."""
    service = getattr(request.app.state, "fraud_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection service is not initialized or still starting up.",
        )
    return service


@app.get(
    "/",
    tags=["General"],
    summary="Root Service Overview",
    description="Returns service metadata, health link, and documentation URLs.",
)
async def root():
    return {
        "service": "Real-Time Fraud Detection API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Service & Model Health Check",
    description="Validates that the XGBoost model and preprocessor are loaded, and returns configuration metadata.",
)
async def health(request: Request):
    service = get_service(request)
    return service.get_health_status()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Score Transaction and Generate SHAP Explanation",
    description="""
Evaluates an incoming raw financial transaction payload:
1. Validates input schema (`Time`, `Amount`, `V1`..`V28`).
2. Applies preprocessing and circadian feature engineering.
3. Generates fraud probability and compares against operational threshold.
4. Categorizes into risk level (`LOW`, `MEDIUM`, `HIGH`).
5. Computes exact SHAP attributions and returns top 5 risk drivers.
6. Persists transaction prediction record to the audit database.
    """,
)
async def predict(
    transaction: TransactionInput,
    request: Request,
    db: Session = Depends(get_db),
):
    service = get_service(request)
    try:
        response = service.predict_and_explain(transaction)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(exc)}",
        )

    # Persist prediction to SQLite audit storage (additive only)
    # If persistence fails, log error gracefully without corrupting or altering the response
    try:
        save_transaction_prediction(
            db=db,
            input_data=transaction,
            prediction=response,
            algorithm=getattr(service, "model_name", "XGBClassifier"),
            version="1.0.0",
        )
    except Exception as db_exc:
        logger.error(f"[Database] Error persisting transaction prediction: {db_exc}", exc_info=True)

    return response


@app.get(
    "/transactions",
    response_model=TransactionListResponse,
    tags=["History & Audit"],
    summary="List Transaction Prediction History",
    description="Returns a paginated list of historically evaluated transactions ordered newest first.",
)
async def list_transactions(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    records, total = get_transactions(db, limit=limit, offset=offset)
    items = [TransactionHistoryItem(**r.to_dict()) for r in records]
    return TransactionListResponse(
        transactions=items,
        count=len(items),
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/transactions/{transaction_id}",
    response_model=TransactionHistoryItem,
    tags=["History & Audit"],
    summary="Get Transaction Prediction by Transaction ID",
    description="Retrieves the stored prediction record for a specific client-provided transaction identifier. If duplicates exist, returns the latest record.",
)
async def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
):
    record = get_transaction_by_id(db, transaction_id=transaction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with transaction_id '{transaction_id}' was not found in audit history.",
        )
    return TransactionHistoryItem(**record.to_dict())


@app.get(
    "/stats",
    response_model=TransactionStatsResponse,
    tags=["History & Audit"],
    summary="Transaction Scoring Statistics",
    description="Returns aggregate counts and fraud rate across stored transaction evaluations.",
)
async def stats(
    db: Session = Depends(get_db),
):
    stats_data = get_transaction_stats(db)
    return TransactionStatsResponse(**stats_data)

