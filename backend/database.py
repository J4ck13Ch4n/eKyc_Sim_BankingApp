from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os

# Check if running in Cloud Run with Cloud SQL settings
# Check if running in Cloud Run with Cloud SQL settings
if os.getenv("DATABASE_URL"):
    # generic connection string (Supabase, Neon, Render, etc.)
    # Note: SQLAlchemy requires 'postgresql://', some providers give 'postgres://'
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(url)
elif os.getenv("INSTANCE_CONNECTION_NAME"):
    db_user = os.environ["DB_USER"]
    db_pass = os.environ["DB_PASSWORD"]
    db_name = os.environ["DB_NAME"]
    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]

    # Google Cloud SQL uses a Unix socket
    socket_path = f"/cloudsql/{instance_connection_name}"
    
    # Construct the database URL for PostgreSQL
    # postgresql+psycopg2://<user>:<password>@/<dbname>?host=/cloudsql/<instance_connection_name>
    SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@/{db_name}?host={socket_path}"
    
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
else:
    # Fallback to Local SQLite
    SQLALCHEMY_DATABASE_URL = "sqlite:///./ekyc.db"
    
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
