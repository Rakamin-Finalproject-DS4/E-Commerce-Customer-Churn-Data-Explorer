"""
Model Training Module for E-Commerce Customer Churn Prediction

This module handles model training, hyperparameter tuning, and evaluation
for customer churn prediction using various machine learning algorithms.

Author: Data Science Team
Version: 1.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Any
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class to handle model training, hyperparameter tuning, and evaluation
    for customer churn prediction.
    
    This class provides methods for:
    - Train-test split
    - Class imbalance handling (SMOTE, Oversampling, Undersampling)
    - Model training
    - Hyperparameter tuning
    - Model evaluation
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the ModelTrainer with configuration parameters.
        
        Args:
            config (Dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_balanced = None
        self.y_train_balanced = None
        self.model = None
        self.best_model = None
        self.predictions = None
        self.evaluation_results = {}

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, 
                    random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            test_size (float): Proportion of test set. Default is 0.3.
            random_state (int): Random state for reproducibility. Default is 42.
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data with test_size={test_size}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"  Training set: {self.X_train.shape}")
        logger.info(f"  Test set: {self.X_test.shape}")
        logger.info(f"  Target distribution (train):")
        logger.info(f"    Class 0: {(self.y_train == 0).sum()} ({(self.y_train == 0).sum()/len(self.y_train)*100:.2f}%)")
        logger.info(f"    Class 1: {(self.y_train == 1).sum()} ({(self.y_train == 1).sum()/len(self.y_train)*100:.2f}%)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def handle_class_imbalance(self, method: str = "SMOTE") -> Tuple:
        """
        Handle class imbalance using SMOTE, RandomOverSampler, or RandomUnderSampler.
        
        Args:
            method (str): Resampling method. Options: "SMOTE", "RandomOverSampler", "RandomUnderSampler".
                         Default is "SMOTE".
            
        Returns:
            Tuple: (X_train_balanced, y_train_balanced)
        """
        logger.info(f"Handling class imbalance using {method}")

        if method == "SMOTE":
            sampler = SMOTE(random_state=self.config.get('random_state', 42))
            logger.info("  Using SMOTE (Synthetic Minority Over-sampling Technique)")
            
        elif method == "RandomOverSampler":
            sampler = RandomOverSampler(random_state=self.config.get('random_state', 42))
            logger.info("  Using RandomOverSampler")
            
        elif method == "RandomUnderSampler":
            sampler = RandomUnderSampler(random_state=self.config.get('random_state', 42), replacement=True)
            logger.info("  Using RandomUnderSampler")
            
        else:
            logger.warning(f"Unknown method {method}. Using SMOTE.")
            sampler = SMOTE(random_state=self.config.get('random_state', 42))

        self.X_train_balanced, self.y_train_balanced = sampler.fit_resample(self.X_train, self.y_train)
        
        logger.info(f"  Balanced dataset shape: {self.X_train_balanced.shape}")
        logger.info(f"  Target distribution (balanced):")
        logger.info(f"    Class 0: {(self.y_train_balanced == 0).sum()}")
        logger.info(f"    Class 1: {(self.y_train_balanced == 1).sum()}")

        return self.X_train_balanced, self.y_train_balanced

    def train_xgboost(self, use_balanced_data: bool = True) -> Any:
        """
        Train XGBoost model.
        
        Args:
            use_balanced_data (bool): Whether to use balanced training data. Default is True.
            
        Returns:
            Trained XGBoost model.
        """
        try:
            from xgboost import XGBClassifier
            
            logger.info("Training XGBoost model...")
            
            hyperparameters = self.config.get('models', {}).get('xgboost', {}).get('hyperparameters', {})
            random_state = self.config.get('models', {}).get('xgboost', {}).get('random_state', 42)
            
            self.model = XGBClassifier(random_state=random_state, **hyperparameters)
            
            X_train = self.X_train_balanced if use_balanced_data else self.X_train
            y_train = self.y_train_balanced if use_balanced_data else self.y_train
            
            self.model.fit(X_train, y_train)
            logger.info("  XGBoost model training completed")
            
            return self.model
            
        except ImportError:
            logger.error("XGBoost not installed. Please install it using: pip install xgboost")
            raise

    def train_random_forest(self, use_balanced_data: bool = True) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            use_balanced_data (bool): Whether to use balanced training data. Default is True.
            
        Returns:
            Trained Random Forest model.
        """
        logger.info("Training Random Forest model...")
        
        hyperparameters = self.config.get('models', {}).get('random_forest', {}).get('hyperparameters', {})
        random_state = self.config.get('models', {}).get('random_forest', {}).get('random_state', 42)
        
        self.model = RandomForestClassifier(random_state=random_state, **hyperparameters)
        
        X_train = self.X_train_balanced if use_balanced_data else self.X_train
        y_train = self.y_train_balanced if use_balanced_data else self.y_train
        
        self.model.fit(X_train, y_train)
        logger.info("  Random Forest model training completed")
        
        return self.model

    def hyperparameter_tuning(self, model_name: str, hyperparameters: Dict, 
                             cv: int = 5, scoring: str = "recall") -> Any:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            model_name (str): Name of the model. Options: "xgboost", "random_forest", "logistic_regression", etc.
            hyperparameters (Dict): Dictionary of hyperparameters to search.
            cv (int): Number of cross-validation folds. Default is 5.
            scoring (str): Scoring metric for tuning. Default is "recall".
            
        Returns:
            Tuned model.
        """
        logger.info(f"Starting hyperparameter tuning for {model_name} (scoring: {scoring})...")
        
        # Initialize base model
        if model_name.lower() == "xgboost":
            try:
                from xgboost import XGBClassifier
                base_model = XGBClassifier(random_state=42)
            except ImportError:
                logger.error("XGBoost not installed")
                raise
                
        elif model_name.lower() == "random_forest":
            base_model = RandomForestClassifier(random_state=42)
            
        elif model_name.lower() == "logistic_regression":
            base_model = LogisticRegression(random_state=42)
            
        else:
            logger.error(f"Unknown model: {model_name}")
            raise ValueError(f"Unknown model: {model_name}")

        # Randomized search
        search = RandomizedSearchCV(
            base_model,
            hyperparameters,
            cv=cv,
            random_state=42,
            scoring=scoring,
            n_iter=20,  # Number of combinations to try
            n_jobs=-1
        )

        search.fit(self.X_train_balanced, self.y_train_balanced)
        
        logger.info(f"  Best parameters found:")
        for param, value in search.best_params_.items():
            logger.info(f"    {param}: {value}")
        logger.info(f"  Best {scoring} score: {search.best_score_:.4f}")

        self.best_model = search.best_estimator_
        return self.best_model

    def predict(self, X_test: Optional[pd.DataFrame] = None, use_best_model: bool = True) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test (pd.DataFrame, optional): Test feature matrix. If None, uses stored X_test.
            use_best_model (bool): Whether to use best tuned model. Default is True.
            
        Returns:
            np.ndarray: Predictions.
        """
        if X_test is None:
            X_test = self.X_test

        model = self.best_model if use_best_model else self.model
        
        if model is None:
            logger.error("No model available for prediction. Train a model first.")
            raise ValueError("No model available")

        self.predictions = model.predict(X_test)
        logger.info(f"Predictions made on {len(self.predictions)} samples")
        
        return self.predictions

    def evaluate_model(self, y_true: Optional[np.ndarray] = None, 
                      y_pred: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true (np.ndarray, optional): True labels. If None, uses stored y_test.
            y_pred (np.ndarray, optional): Predicted labels. If None, uses stored predictions.
            
        Returns:
            Dict: Dictionary containing evaluation metrics.
        """
        if y_true is None:
            y_true = self.y_test
        
        if y_pred is None:
            if self.predictions is None:
                logger.error("No predictions available. Run predict() first.")
                raise ValueError("No predictions available")
            y_pred = self.predictions

        logger.info("Evaluating model performance...")

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # ROC AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = None
            logger.warning("Could not calculate ROC AUC")

        self.evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }

        # Log results
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            logger.info(f"  ROC AUC:   {roc_auc:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TP: {tp}, FP: {fp}")
        logger.info(f"    FN: {fn}, TN: {tn}")

        return self.evaluation_results

    def get_feature_importance(self, top_n: int = 25) -> pd.Series:
        """
        Get feature importance from tree-based models.
        
        Args:
            top_n (int): Number of top features to return. Default is 25.
            
        Returns:
            pd.Series: Feature importance scores (sorted).
        """
        model = self.best_model if self.best_model else self.model
        
        if model is None:
            logger.error("No model available")
            raise ValueError("No model available")

        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None

        feature_names = self.X_train.columns
        importances = model.feature_importances_
        
        feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        logger.info(f"Top {top_n} important features:")
        for i, (feature, importance) in enumerate(feature_importance.head(top_n).items(), 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")

        return feature_importance

    def get_classification_report(self, y_true: Optional[np.ndarray] = None,
                                 y_pred: Optional[np.ndarray] = None) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true (np.ndarray, optional): True labels.
            y_pred (np.ndarray, optional): Predicted labels.
            
        Returns:
            str: Classification report.
        """
        if y_true is None:
            y_true = self.y_test
            
        if y_pred is None:
            y_pred = self.predictions

        report = classification_report(y_true, y_pred)
        logger.info("Classification Report:")
        logger.info(report)
        
        return report


def complete_training_pipeline(config_dict: Dict, X: pd.DataFrame, y: pd.Series,
                               model_name: str = "xgboost") -> Tuple[Any, Dict]:
    """
    Complete model training pipeline.
    
    This function orchestrates the entire training workflow:
    1. Prepare data (train-test split)
    2. Handle class imbalance (SMOTE)
    3. Train model
    4. Hyperparameter tuning
    5. Evaluate model
    
    Args:
        config_dict (Dict): Configuration dictionary.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        model_name (str): Name of model to train. Default is "xgboost".
        
    Returns:
        Tuple: (trained_model, evaluation_results)
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = ModelTrainer(config=config_dict)

    # Prepare data
    trainer.prepare_data(
        X, y,
        test_size=config_dict.get('train_test_split', {}).get('test_size', 0.3),
        random_state=config_dict.get('train_test_split', {}).get('random_state', 42)
    )

    # Handle class imbalance
    trainer.handle_class_imbalance(
        method=config_dict.get('imbalance_handling', {}).get('method', 'SMOTE')
    )

    # Train model
    if model_name.lower() == "xgboost":
        trainer.train_xgboost(use_balanced_data=True)
    elif model_name.lower() == "random_forest":
        trainer.train_random_forest(use_balanced_data=True)
    else:
        logger.error(f"Unknown model: {model_name}")
        raise ValueError(f"Unknown model: {model_name}")

    # Make predictions
    trainer.predict(use_best_model=False)

    # Evaluate
    evaluation_results = trainer.evaluate_model()

    # Feature importance
    try:
        feature_importance = trainer.get_feature_importance()
    except:
        logger.warning("Could not extract feature importance")
        feature_importance = None

    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE COMPLETED")
    logger.info("=" * 60)

    return trainer.model, evaluation_results
