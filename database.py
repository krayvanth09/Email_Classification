"""
Database operations for email classification system
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, EmailClassification, TrainingData, ModelMetrics
import pandas as pd
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")
    
    def save_classification_result(self, original_email: str, masked_email: str, 
                                 classification: str, pii_entities: List[Dict]) -> int:
        """Save email classification result to database"""
        session = self.Session()
        try:
            result = EmailClassification(
                original_email=original_email,
                masked_email=masked_email,
                classification=classification,
                pii_entities=pii_entities
            )
            session.add(result)
            session.commit()
            result_id = result.id
            logger.info(f"Saved classification result with ID: {result_id}")
            return result_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving classification result: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_training_data_to_db(self, csv_path: str) -> int:
        """Load training data from CSV to database"""
        session = self.Session()
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} training samples to database")
            
            # Clear existing training data
            session.query(TrainingData).delete()
            
            # Insert new data
            count = 0
            for _, row in df.iterrows():
                training_data = TrainingData(
                    email_text=row['email'],
                    true_classification=row['type']
                )
                session.add(training_data)
                count += 1
                
                # Commit in batches for better performance
                if count % 1000 == 0:
                    session.commit()
                    logger.info(f"Loaded {count} samples...")
            
            session.commit()
            logger.info(f"Successfully loaded {count} training samples to database")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading training data: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_training_data(self) -> pd.DataFrame:
        """Get training data from database as DataFrame"""
        session = self.Session()
        try:
            training_data = session.query(TrainingData).all()
            
            data = []
            for item in training_data:
                data.append({
                    'email': item.email_text,
                    'type': item.true_classification
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} training samples from database")
            return df
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            raise
        finally:
            session.close()
    
    def save_model_metrics(self, version: str, metrics: Dict[str, Any]) -> int:
        """Save model performance metrics"""
        session = self.Session()
        try:
            model_metrics = ModelMetrics(
                model_version=version,
                accuracy=str(metrics.get('accuracy', '')),
                precision=str(metrics.get('precision', '')),
                recall=str(metrics.get('recall', '')),
                f1_score=str(metrics.get('f1_score', '')),
                training_samples=metrics.get('training_samples', 0)
            )
            session.add(model_metrics)
            session.commit()
            metrics_id = model_metrics.id
            logger.info(f"Saved model metrics with ID: {metrics_id}")
            return metrics_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving model metrics: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_classification_history(self, limit: int = 100) -> List[Dict]:
        """Get recent classification history"""
        session = self.Session()
        try:
            results = session.query(EmailClassification)\
                           .order_by(EmailClassification.processed_at.desc())\
                           .limit(limit).all()
            
            history = []
            for result in results:
                history.append({
                    'id': result.id,
                    'classification': result.classification,
                    'processed_at': result.processed_at.isoformat(),
                    'pii_count': len(result.pii_entities) if result.pii_entities else 0
                })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving classification history: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        session = self.Session()
        try:
            total_classifications = session.query(EmailClassification).count()
            
            # Get classification distribution
            from sqlalchemy import func
            distribution = session.query(
                EmailClassification.classification,
                func.count(EmailClassification.id)
            ).group_by(EmailClassification.classification).all()
            
            stats = {
                'total_classifications': total_classifications,
                'classification_distribution': {item[0]: item[1] for item in distribution}
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error retrieving classification stats: {str(e)}")
            raise
        finally:
            session.close()