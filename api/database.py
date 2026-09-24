"""
Database configuration, session management, and table initialization.
Supports SQLite storage with configurable DATABASE_URL for testing isolation.
"""
import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Base directory for the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "fraud_detection.db"

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
)

# SQLite engine configuration (check_same_thread=False allows multi-threaded FastAPI access)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db(engine_override=None):
    """
    Idempotently creates database tables if they do not exist.
    Ensures target directory exists when using SQLite file paths.
    Does not drop or overwrite existing tables/data.
    """
    active_engine = engine_override or engine
    
    # If using local file SQLite, ensure directory exists
    db_url_str = str(active_engine.url)
    if "sqlite:///" in db_url_str and ":memory:" not in db_url_str:
        file_path_str = db_url_str.replace("sqlite:///", "")
        db_file = Path(file_path_str)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=active_engine)
    print(f"[Database] Initialized tables successfully with engine: {active_engine.url}")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a database session per request.
    Ensures proper session closure upon request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
