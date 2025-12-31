"""
Database configuration and session management for ThermoSense
Uses SQLAlchemy with PostgreSQL backend
"""
import os
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging

logger = logging.getLogger(__name__)

# Database URL from environment or default to SQLite for development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./thermosense.db"  # Fallback to SQLite for easy development
)

# For PostgreSQL with pgvector in production:
# DATABASE_URL = "postgresql://user:password@localhost:5432/thermosense"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=False,  # Set to True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for all models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints to get database session
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables
    Call this on application startup
    """
    from backend.db import models  # Import models to register them
    
    logger.info(f"Initializing database: {DATABASE_URL}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def check_db_connection() -> bool:
    """
    Check if database connection is working
    Returns True if connection successful, False otherwise
    """
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# For PostgreSQL with pgvector extension
def enable_pgvector(connection, connection_record):
    """Enable pgvector extension on connection"""
    cursor = connection.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.commit()
    except Exception as e:
        logger.warning(f"Could not enable pgvector extension: {e}")
    finally:
        cursor.close()


# Register pgvector enabler for PostgreSQL
if "postgresql" in DATABASE_URL:
    event.listen(engine, "connect", enable_pgvector)
