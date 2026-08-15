from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.config import settings

# The engine is the actual connection to PostgreSQL
# Think of it as the phone line between Python and the database
engine = create_engine(settings.DATABASE_URL)

# A session is one conversation with the database
# SessionLocal is a factory — it creates a new conversation whenever we need one
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is the parent class for all our database tables
# Every table we create later will inherit from this
Base = declarative_base()

def get_db():
    """Opens a database session, gives it to the API, then closes it when done."""
    db = SessionLocal()
    try:
        yield db    # Hand the session to whoever asked for it
    finally:
        db.close()  # Always close, even if an error happened