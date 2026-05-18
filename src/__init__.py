"""
E-Commerce Customer Churn Prediction - Source Code Package

This package contains all modular code for data preprocessing, feature engineering,
and model training for customer churn prediction.

Modules:
    data_preprocessing: Data cleaning and preparation
    feature_engineering: Feature transformation and encoding
    model_trainer: Model training and evaluation

Author: Data Science Team
Version: 1.0
"""

from src.data_preprocessing import DataPreprocessor, preprocess_pipeline
from src.feature_engineering import FeatureEngineer, feature_engineering_pipeline
from src.model_trainer import ModelTrainer, complete_training_pipeline

__version__ = "1.0.0"
__all__ = [
    "DataPreprocessor",
    "preprocess_pipeline",
    "FeatureEngineer",
    "feature_engineering_pipeline",
    "ModelTrainer",
    "complete_training_pipeline"
]
