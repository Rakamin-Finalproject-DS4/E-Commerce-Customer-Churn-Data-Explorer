"""
Save Artifacts Module for E-Commerce Customer Churn Prediction

This module handles saving trained models, scalers, encoders, and other
artifacts for later use in production environments.

Author: Data Science Team
Version: 1.0
"""

import os
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArtifactManager:
    """
    A class to manage saving and loading of model artifacts.
    
    This class provides methods for:
    - Saving trained models
    - Saving scalers and encoders
    - Saving feature lists
    - Saving model metadata
    - Loading artifacts for inference
    """

    def __init__(self, artifacts_dir: str = "models"):
        """
        Initialize the ArtifactManager.
        
        Args:
            artifacts_dir (str): Directory to save artifacts. Default is "models".
        """
        self.artifacts_dir = artifacts_dir
        self._ensure_directory_exists(artifacts_dir)
        logger.info(f"Artifact manager initialized with directory: {artifacts_dir}")

    def _ensure_directory_exists(self, directory: str):
        """
        Ensure that the directory exists, create if it doesn't.
        
        Args:
            directory (str): Path to directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    def save_model(self, model: Any, model_name: str, model_type: str = "xgboost",
                   overwrite: bool = True) -> str:
        """
        Save trained model using joblib or pickle.
        
        This function saves the trained machine learning model to disk using joblib
        (preferred for sklearn models) or pickle (fallback).
        
        Args:
            model: Trained model object.
            model_name (str): Name for the model (without extension).
                             Example: "xgboost_model", "rf_model"
            model_type (str): Type of model. Default is "xgboost".
            overwrite (bool): Whether to overwrite existing file. Default is True.
            
        Returns:
            str: Path to saved model.
            
        Example:
            >>> artifact_manager = ArtifactManager()
            >>> saved_path = artifact_manager.save_model(
            ...     model=trained_xgb_model,
            ...     model_name="xgboost_churn_v1",
            ...     model_type="xgboost"
            ... )
        """
        model_path = os.path.join(self.artifacts_dir, f"{model_name}.pkl")

        if os.path.exists(model_path) and not overwrite:
            logger.warning(f"Model file already exists: {model_path}. Skipping save.")
            return model_path

        try:
            # Try using joblib first (better for sklearn models)
            try:
                import joblib
                joblib.dump(model, model_path)
                logger.info(f"Model saved using joblib: {model_path}")
            except ImportError:
                # Fallback to pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved using pickle: {model_path}")

            logger.info(f"✓ Successfully saved {model_type} model to {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"✗ Failed to save model: {str(e)}")
            raise

    def save_scaler(self, scaler: Any, scaler_name: str = "scaler",
                   overwrite: bool = True) -> str:
        """
        Save fitted scaler for feature scaling.
        
        This function saves the fitted scaler (StandardScaler, MinMaxScaler, etc.)
        to disk for use during inference on new data.
        
        Args:
            scaler: Fitted scaler object (e.g., StandardScaler, MinMaxScaler).
            scaler_name (str): Name for the scaler. Default is "scaler".
            overwrite (bool): Whether to overwrite existing file. Default is True.
            
        Returns:
            str: Path to saved scaler.
            
        Example:
            >>> from sklearn.preprocessing import StandardScaler
            >>> from sklearn.preprocessing import MinMaxScaler
            >>> 
            >>> # Example with StandardScaler
            >>> scaler = StandardScaler()
            >>> scaler.fit(X_train)
            >>> saved_scaler_path = artifact_manager.save_scaler(
            ...     scaler=scaler,
            ...     scaler_name="standard_scaler_v1"
            ... )
            >>> 
            >>> # Example with MinMaxScaler
            >>> minmax_scaler = MinMaxScaler()
            >>> minmax_scaler.fit(X_train)
            >>> saved_minmax_path = artifact_manager.save_scaler(
            ...     scaler=minmax_scaler,
            ...     scaler_name="minmax_scaler_v1"
            ... )
        """
        scaler_path = os.path.join(self.artifacts_dir, f"{scaler_name}.pkl")

        if os.path.exists(scaler_path) and not overwrite:
            logger.warning(f"Scaler file already exists: {scaler_path}. Skipping save.")
            return scaler_path

        try:
            try:
                import joblib
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved using joblib: {scaler_path}")
            except ImportError:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"Scaler saved using pickle: {scaler_path}")

            logger.info(f"✓ Successfully saved scaler to {scaler_path}")
            return scaler_path

        except Exception as e:
            logger.error(f"✗ Failed to save scaler: {str(e)}")
            raise

    def save_feature_list(self, feature_names: List[str], feature_list_name: str = "features",
                         overwrite: bool = True) -> str:
        """
        Save list of feature names used in the model.
        
        This function saves the feature names to ensure consistency between
        training and inference.
        
        Args:
            feature_names (List[str]): List of feature column names.
            feature_list_name (str): Name for the feature list. Default is "features".
            overwrite (bool): Whether to overwrite existing file. Default is True.
            
        Returns:
            str: Path to saved feature list.
            
        Example:
            >>> feature_names = ['Tenure_std', 'SatisfactionScore', 'OrderCount', 
            ...                  'CashbackAmount_norm', ...]
            >>> saved_features_path = artifact_manager.save_feature_list(
            ...     feature_names=feature_names,
            ...     feature_list_name="model_features_v1"
            ... )
        """
        feature_path = os.path.join(self.artifacts_dir, f"{feature_list_name}.json")

        if os.path.exists(feature_path) and not overwrite:
            logger.warning(f"Feature list file already exists: {feature_path}. Skipping save.")
            return feature_path

        try:
            with open(feature_path, 'w') as f:
                json.dump(feature_names, f, indent=4)

            logger.info(f"✓ Successfully saved {len(feature_names)} feature names to {feature_path}")
            return feature_path

        except Exception as e:
            logger.error(f"✗ Failed to save feature list: {str(e)}")
            raise

    def save_model_metadata(self, metadata: Dict, metadata_name: str = "model_metadata",
                           overwrite: bool = True) -> str:
        """
        Save model metadata (metrics, hyperparameters, etc.).
        
        This function saves important metadata about the model including:
        - Training and evaluation metrics
        - Hyperparameters
        - Feature importance
        - Training date and time
        
        Args:
            metadata (Dict): Dictionary containing metadata.
            metadata_name (str): Name for metadata file. Default is "model_metadata".
            overwrite (bool): Whether to overwrite existing file. Default is True.
            
        Returns:
            str: Path to saved metadata file.
            
        Example:
            >>> metadata = {
            ...     'model_type': 'XGBoost',
            ...     'accuracy': 0.92,
            ...     'precision': 0.89,
            ...     'recall': 0.86,
            ...     'f1_score': 0.875,
            ...     'hyperparameters': {
            ...         'max_depth': 6,
            ...         'learning_rate': 0.1,
            ...         'n_estimators': 100
            ...     },
            ...     'training_date': '2024-05-10',
            ...     'training_samples': 3941
            ... }
            >>> saved_metadata_path = artifact_manager.save_model_metadata(
            ...     metadata=metadata,
            ...     metadata_name="xgboost_metadata_v1"
            ... )
        """
        metadata_path = os.path.join(self.artifacts_dir, f"{metadata_name}.json")

        if os.path.exists(metadata_path) and not overwrite:
            logger.warning(f"Metadata file already exists: {metadata_path}. Skipping save.")
            return metadata_path

        try:
            # Add timestamp if not present
            if 'saved_timestamp' not in metadata:
                metadata['saved_timestamp'] = datetime.now().isoformat()

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"✓ Successfully saved model metadata to {metadata_path}")
            return metadata_path

        except Exception as e:
            logger.error(f"✗ Failed to save metadata: {str(e)}")
            raise

    def save_all_artifacts(self, model: Any, scaler: Any, feature_names: List[str],
                          metadata: Dict, model_name: str = "final_model",
                          overwrite: bool = True) -> Dict[str, str]:
        """
        Save all model artifacts at once (convenience function).
        
        This function saves model, scaler, features, and metadata in one call.
        
        Args:
            model: Trained model object.
            scaler: Fitted scaler object.
            feature_names (List[str]): List of feature names.
            metadata (Dict): Model metadata dictionary.
            model_name (str): Base name for artifacts. Default is "final_model".
            overwrite (bool): Whether to overwrite existing files. Default is True.
            
        Returns:
            Dict: Dictionary with paths to all saved artifacts.
            
        Example:
            >>> paths = artifact_manager.save_all_artifacts(
            ...     model=trained_model,
            ...     scaler=fitted_scaler,
            ...     feature_names=X.columns.tolist(),
            ...     metadata={
            ...         'model_type': 'XGBoost',
            ...         'accuracy': 0.92,
            ...         'precision': 0.89,
            ...         'recall': 0.86,
            ...         'f1_score': 0.875
            ...     },
            ...     model_name="xgboost_churn_final"
            ... )
            >>> print(paths)
            {'model_path': 'models/xgboost_churn_final.pkl',
             'scaler_path': 'models/xgboost_churn_final_scaler.pkl',
             'features_path': 'models/xgboost_churn_final_features.json',
             'metadata_path': 'models/xgboost_churn_final_metadata.json'}
        """
        logger.info("=" * 60)
        logger.info("SAVING ALL MODEL ARTIFACTS")
        logger.info("=" * 60)

        paths = {}

        # Save model
        model_path = self.save_model(model, model_name, overwrite=overwrite)
        paths['model_path'] = model_path

        # Save scaler
        scaler_path = self.save_scaler(scaler, f"{model_name}_scaler", overwrite=overwrite)
        paths['scaler_path'] = scaler_path

        # Save features
        features_path = self.save_feature_list(feature_names, f"{model_name}_features", overwrite=overwrite)
        paths['features_path'] = features_path

        # Save metadata
        metadata_path = self.save_model_metadata(metadata, f"{model_name}_metadata", overwrite=overwrite)
        paths['metadata_path'] = metadata_path

        logger.info("=" * 60)
        logger.info("ALL ARTIFACTS SAVED SUCCESSFULLY")
        logger.info("=" * 60)

        return paths

    def load_model(self, model_path: str) -> Any:
        """
        Load saved model from disk.
        
        Args:
            model_path (str): Path to saved model file.
            
        Returns:
            Loaded model object.
        """
        try:
            try:
                import joblib
                model = joblib.load(model_path)
                logger.info(f"Model loaded using joblib: {model_path}")
            except ImportError:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model loaded using pickle: {model_path}")

            logger.info(f"✓ Successfully loaded model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise

    def load_scaler(self, scaler_path: str) -> Any:
        """
        Load saved scaler from disk.
        
        Args:
            scaler_path (str): Path to saved scaler file.
            
        Returns:
            Loaded scaler object.
        """
        try:
            try:
                import joblib
                scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded using joblib: {scaler_path}")
            except ImportError:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"Scaler loaded using pickle: {scaler_path}")

            logger.info(f"✓ Successfully loaded scaler from {scaler_path}")
            return scaler

        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {str(e)}")
            raise

    def load_feature_list(self, feature_path: str) -> List[str]:
        """
        Load saved feature list from disk.
        
        Args:
            feature_path (str): Path to saved feature JSON file.
            
        Returns:
            List of feature names.
        """
        try:
            with open(feature_path, 'r') as f:
                features = json.load(f)

            logger.info(f"✓ Successfully loaded {len(features)} features from {feature_path}")
            return features

        except Exception as e:
            logger.error(f"✗ Failed to load features: {str(e)}")
            raise

    def load_metadata(self, metadata_path: str) -> Dict:
        """
        Load saved metadata from disk.
        
        Args:
            metadata_path (str): Path to saved metadata JSON file.
            
        Returns:
            Dictionary containing metadata.
        """
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            logger.info(f"✓ Successfully loaded metadata from {metadata_path}")
            return metadata

        except Exception as e:
            logger.error(f"✗ Failed to load metadata: {str(e)}")
            raise

    def list_artifacts(self) -> List[str]:
        """
        List all saved artifacts in the artifacts directory.
        
        Returns:
            List of file names in the artifacts directory.
        """
        try:
            if not os.path.exists(self.artifacts_dir):
                logger.warning(f"Artifacts directory not found: {self.artifacts_dir}")
                return []

            artifacts = os.listdir(self.artifacts_dir)
            logger.info(f"Found {len(artifacts)} artifacts:")
            for artifact in artifacts:
                file_path = os.path.join(self.artifacts_dir, artifact)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {artifact} ({file_size:,.0f} bytes)")

            return artifacts

        except Exception as e:
            logger.error(f"✗ Failed to list artifacts: {str(e)}")
            raise
