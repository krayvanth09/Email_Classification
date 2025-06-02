"""
Email Classification Module
Handles training and inference for email classification
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EmailClassifier:
    def __init__(self, model_path: str = "email_classifier_model.pkl"):
        """
        Initialize Email Classifier
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.classes = ['Incident', 'Request', 'Change', 'Problem']
        
        # Try to load existing model
        self._load_model()
        
        # If no model exists, train a new one
        if self.model is None:
            logger.info("No existing model found. Training new model...")
            self.train_model()
    
    def _load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.info(f"Model file {self.model_path} not found")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess email text for classification
        
        Args:
            text: Raw email text
            
        Returns:
            Preprocessed text
        """
        # Basic text cleaning
        text = text.lower()
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train_model(self, data_path: str = "Email_Classification_DataSet.csv"):
        """
        Train the email classification model
        
        Args:
            data_path: Path to training data CSV file
        """
        try:
            logger.info("Starting model training...")
            
            # Load training data
            if not os.path.exists(data_path):
                logger.error(f"Training data file {data_path} not found")
                # Create a simple fallback model
                self._create_fallback_model()
                return
            
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} training samples")
            
            # Preprocess data
            df['processed_email'] = df['email'].apply(self.preprocess_text)
            
            # Split data
            X = df['processed_email']
            y = df['type']
            
            # Ensure we have enough samples for each class
            min_class_count = y.value_counts().min()
            if min_class_count < 2:
                logger.warning(f"Not enough samples for proper train/test split. Min class count: {min_class_count}")
                # Use a smaller test size or no split for very small datasets
                if len(df) < 10:
                    X_train, X_test, y_train, y_test = X, X, y, y
                else:
                    test_size = min(0.2, max(1/len(df), 2/min_class_count))
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            # Create pipeline with TF-IDF and Logistic Regression
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model when training data is not available"""
        logger.info("Creating fallback model...")
        
        # Create a simple pipeline that can still classify
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Create minimal training data based on keywords
        fallback_data = [
            ("urgent problem system crashed error", "Incident"),
            ("request information help support", "Request"),
            ("change update modify configuration", "Change"),
            ("problem issue recurring persistent", "Problem"),
            ("incident critical down failure", "Incident"),
            ("please provide assistance guidance", "Request"),
            ("update upgrade modification", "Change"),
            ("ongoing problem continues", "Problem")
        ]
        
        X_fallback = [text for text, _ in fallback_data]
        y_fallback = [label for _, label in fallback_data]
        
        self.model.fit(X_fallback, y_fallback)
        self._save_model()
        
        logger.info("Fallback model created and saved")
    
    def classify(self, email_text: str) -> str:
        """
        Classify an email into one of the predefined categories
        
        Args:
            email_text: Email text to classify
            
        Returns:
            Predicted category
        """
        try:
            if self.model is None:
                logger.error("No model available for classification")
                return "Request"  # Default category
            
            # Preprocess text
            processed_text = self.preprocess_text(email_text)
            
            # Make prediction
            prediction = self.model.predict([processed_text])[0]
            
            logger.debug(f"Classified email as: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            return "Request"  # Default category on error
    
    def get_prediction_probabilities(self, email_text: str) -> Dict[str, float]:
        """
        Get prediction probabilities for all classes
        
        Args:
            email_text: Email text to classify
            
        Returns:
            Dictionary mapping class names to probabilities
        """
        try:
            if self.model is None:
                return {cls: 0.25 for cls in self.classes}
            
            processed_text = self.preprocess_text(email_text)
            probabilities = self.model.predict_proba([processed_text])[0]
            
            # Get class names from the model
            classes = self.model.named_steps['classifier'].classes_
            
            return dict(zip(classes, probabilities))
            
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {str(e)}")
            return {cls: 0.25 for cls in self.classes}
