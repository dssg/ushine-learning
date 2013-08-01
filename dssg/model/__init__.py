from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Connect to the database
engine = create_engine()

# Base class for declarative class mapping
Base = declarative_base()

# Create session
Session = sessionmaker(bind=engine)

__all__ = ['Base', 'Session']