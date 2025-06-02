"""
Database models for email classification system
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class EmailClassification(Base):
    """Model for storing email classification results"""
    __tablename__ = 'email_classifications'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    original_email = Column(Text, nullable=False)
    masked_email = Column(Text, nullable=False)
    classification = Column(String(50), nullable=False)
    pii_entities = Column(JSON, nullable=True)  # Store detected PII entities as JSON
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<EmailClassification(id={self.id}, classification='{self.classification}', processed_at='{self.processed_at}')>"

class TrainingData(Base):
    """Model for storing training dataset"""
    __tablename__ = 'training_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email_text = Column(Text, nullable=False)
    true_classification = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_validated = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<TrainingData(id={self.id}, classification='{self.true_classification}')>"

class ModelMetrics(Base):
    """Model for storing model performance metrics"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    accuracy = Column(String(10), nullable=True)
    precision = Column(String(10), nullable=True)
    recall = Column(String(10), nullable=True)
    f1_score = Column(String(10), nullable=True)
    training_samples = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelMetrics(id={self.id}, version='{self.model_version}', accuracy='{self.accuracy}')>"

# Database setup
def get_database_url():
    """Get database URL from environment"""
    return os.getenv('DATABASE_URL', 'postgresql://localhost/email_classification')

def create_tables():
    """Create all database tables"""
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """Get database session"""
    engine = create_engine(get_database_url())
    Session = sessionmaker(bind=engine)
    return Session()