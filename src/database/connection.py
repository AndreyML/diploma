"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config.settings import settings
from src.database.models import Base
from typing import Generator, AsyncGenerator
from contextlib import asynccontextmanager
# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,
    pool_recycle=300
)
# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[Session, None]:
    """Async context manager for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
async def init_db():
    """Initialize database - create tables if they don't exist"""
    create_tables()
