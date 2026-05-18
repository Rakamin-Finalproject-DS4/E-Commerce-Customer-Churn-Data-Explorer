"""
Feature Engineering Module for E-Commerce Customer Churn Prediction

This module handles feature transformation, encoding, scaling, and extraction
to prepare features for machine learning model training.

Author: Data Science Team
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import logging
from typing import Tuple, List, Dict, Optional
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class to handle feature engineering tasks for customer churn prediction.
    
    This class provides methods for:
    - Feature transformation (normalization, standardization)
    - Feature encoding (label encoding, one-hot encoding)
    - Feature extraction (creating new features)
    - Feature selection (dropping irrelevant features)
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the FeatureEngineer with configuration parameters.
        
        Args:
            config (Dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self.df = None
        self.scalers = {}  # Store scalers for later use
        self.encoders = {}  # Store encoders for later use

    def set_data(self, df: pd.DataFrame):
        """
        Set the dataframe to work with.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        """
        self.df = df.copy()
        logger.info(f"Data set for feature engineering. Shape: {self.df.shape}")

    def normalize_features(self, features: List[str], column_suffix: str = "_norm") -> pd.DataFrame:
        """
        Normalize numerical features to [0, 1] range using MinMaxScaler.
        
        Normalization is useful for features with different scales and preserves
        the distribution of the original data.
        
        Args:
            features (List[str]): List of feature names to normalize.
            column_suffix (str): Suffix to add to normalized column names.
            
        Returns:
            pd.DataFrame: Dataframe with normalized features added.
        """
        logger.info(f"Normalizing features: {features}")

        for feature in features:
            if feature in self.df.columns:
                scaler = MinMaxScaler()
                normalized_values = scaler.fit_transform(self.df[[feature]])
                new_col_name = f"{feature}{column_suffix}"
                self.df[new_col_name] = normalized_values
                self.scalers[new_col_name] = scaler
                logger.info(f"  {feature} → {new_col_name}")
            else:
                logger.warning(f"Feature {feature} not found in dataframe")

        return self.df

    def standardize_features(self, features: List[str], column_suffix: str = "_std") -> pd.DataFrame:
        """
        Standardize numerical features using StandardScaler (mean=0, std=1).
        
        Standardization is useful for features with skewed distributions
        and improves model performance for distance-based algorithms.
        
        Args:
            features (List[str]): List of feature names to standardize.
            column_suffix (str): Suffix to add to standardized column names.
            
        Returns:
            pd.DataFrame: Dataframe with standardized features added.
        """
        logger.info(f"Standardizing features: {features}")

        for feature in features:
            if feature in self.df.columns:
                scaler = StandardScaler()
                standardized_values = scaler.fit_transform(self.df[[feature]])
                new_col_name = f"{feature}{column_suffix}"
                self.df[new_col_name] = standardized_values
                self.scalers[new_col_name] = scaler
                logger.info(f"  {feature} → {new_col_name}")
            else:
                logger.warning(f"Feature {feature} not found in dataframe")

        return self.df

    def label_encode_features(self, features: Dict[str, Dict], inplace: bool = True) -> pd.DataFrame:
        """
        Label encode categorical features.
        
        Maps categorical values to numerical values based on provided mapping.
        Example: Gender: {'Female': 0, 'Male': 1}
        
        Args:
            features (Dict): Dictionary with column names as keys and mapping dicts as values.
            inplace (bool): If True, replace original columns. Default is True.
            
        Returns:
            pd.DataFrame: Dataframe with encoded features.
        """
        logger.info(f"Label encoding features: {list(features.keys())}")

        for feature, mapping in features.items():
            if feature in self.df.columns:
                self.df[feature] = self.df[feature].map(mapping)
                self.encoders[feature] = mapping
                logger.info(f"  {feature}: Mapped with values {mapping}")
            else:
                logger.warning(f"Feature {feature} not found in dataframe")

        return self.df

    def onehot_encode_features(self, features: List[str], drop_original: bool = True) -> pd.DataFrame:
        """
        One-hot encode categorical features.
        
        Creates binary columns for each category in the specified features.
        
        Args:
            features (List[str]): List of feature names to one-hot encode.
            drop_original (bool): If True, drop original columns. Default is True.
            
        Returns:
            pd.DataFrame: Dataframe with one-hot encoded features.
        """
        logger.info(f"One-hot encoding features: {features}")

        for feature in features:
            if feature in self.df.columns:
                onehot = pd.get_dummies(self.df[feature], prefix=feature, drop_first=False)
                self.df = self.df.join(onehot)
                
                if drop_original:
                    self.df = self.df.drop(columns=[feature])
                
                logger.info(f"  {feature}: Created {len(onehot.columns)} binary columns")
            else:
                logger.warning(f"Feature {feature} not found in dataframe")

        return self.df

    def extract_customer_category(self, order_count_col: str = "OrderCount", 
                                 inplace: bool = True) -> pd.DataFrame:
        """
        Extract customer category feature based on OrderCount.
        
        Categories:
        - Bronze: OrderCount < 4
        - Silver: 4 <= OrderCount < 9
        - Gold: OrderCount >= 9
        
        Args:
            order_count_col (str): Name of the OrderCount column.
            inplace (bool): If True, add to existing dataframe. Default is True.
            
        Returns:
            pd.DataFrame: Dataframe with new CustomerCategory feature.
        """
        if order_count_col not in self.df.columns:
            logger.warning(f"Column {order_count_col} not found for category extraction")
            return self.df

        logger.info(f"Extracting CustomerCategory from {order_count_col}")

        def categorize(order_count):
            if order_count < 4:
                return 'Bronze'
            elif order_count < 9:
                return 'Silver'
            else:
                return 'Gold'

        customer_category = self.df[order_count_col].apply(categorize)
        
        # Map to numerical values
        category_mapping = {'Bronze': 0, 'Silver': 1, 'Gold': 2}
        customer_category_encoded = customer_category.map(category_mapping)
        
        self.df['CustomerCategory'] = customer_category_encoded
        self.encoders['CustomerCategory'] = category_mapping
        
        logger.info(f"  Created CustomerCategory feature with distribution:")
        for cat, val in category_mapping.items():
            count = (customer_category == cat).sum()
            logger.info(f"    {cat}: {count} customers")

        return self.df

    def extract_satisfaction_category(self, satisfaction_col: str = "SatisfactionScore") -> pd.DataFrame:
        """
        Extract satisfaction category from satisfaction score.
        
        Categories:
        - VeryDissatisfied: Score 1-1.5
        - Dissatisfied: Score 1.5-2.5
        - Neutral: Score 2.5-3.5
        - Satisfied: Score 3.5-4.5
        - VerySatisfied: Score 4.5-5
        
        Args:
            satisfaction_col (str): Name of the satisfaction score column.
            
        Returns:
            pd.DataFrame: Dataframe with new SatisfactionCategory feature.
        """
        if satisfaction_col not in self.df.columns:
            logger.warning(f"Column {satisfaction_col} not found for category extraction")
            return self.df

        logger.info(f"Extracting SatisfactionCategory from {satisfaction_col}")

        def categorize_satisfaction(score):
            if score <= 1.5:
                return 'VeryDissatisfied'
            elif score <= 2.5:
                return 'Dissatisfied'
            elif score <= 3.5:
                return 'Neutral'
            elif score <= 4.5:
                return 'Satisfied'
            else:
                return 'VerySatisfied'

        satisfaction_category = self.df[satisfaction_col].apply(categorize_satisfaction)
        
        # Map to numerical values
        satisfaction_mapping = {
            'VeryDissatisfied': 0,
            'Dissatisfied': 1,
            'Neutral': 2,
            'Satisfied': 3,
            'VerySatisfied': 4
        }
        satisfaction_category_encoded = satisfaction_category.map(satisfaction_mapping)
        
        self.df['SatisfactionCategory'] = satisfaction_category_encoded
        self.encoders['SatisfactionCategory'] = satisfaction_mapping
        
        logger.info(f"  Created SatisfactionCategory feature")

        return self.df

    def drop_features(self, features: List[str]) -> pd.DataFrame:
        """
        Drop irrelevant or redundant features from the dataframe.
        
        Args:
            features (List[str]): List of feature names to drop.
            
        Returns:
            pd.DataFrame: Dataframe with dropped features.
        """
        logger.info(f"Dropping features: {features}")
        
        features_to_drop = [f for f in features if f in self.df.columns]
        self.df = self.df.drop(columns=features_to_drop)
        
        logger.info(f"  Dropped {len(features_to_drop)} features")
        logger.info(f"  Remaining shape: {self.df.shape}")

        return self.df

    def get_feature_info(self) -> Dict:
        """
        Get information about current features.
        
        Returns:
            Dict: Dictionary containing feature statistics.
        """
        info = {
            'total_features': self.df.shape[1],
            'numerical_features': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': self.df.select_dtypes(include=['object']).columns.tolist(),
            'shape': self.df.shape
        }
        return info

    def get_engineered_data(self) -> pd.DataFrame:
        """
        Get the feature-engineered dataframe.
        
        Returns:
            pd.DataFrame: The feature-engineered dataframe.
        """
        return self.df

    def save_scalers(self, output_path: str):
        """
        Save the scalers and encoders for later use in production.
        
        Args:
            output_path (str): Path to save the pickle file.
        """
        with open(output_path, 'wb') as f:
            pickle.dump({'scalers': self.scalers, 'encoders': self.encoders}, f)
        logger.info(f"Scalers and encoders saved to {output_path}")

    def load_scalers(self, input_path: str):
        """
        Load previously saved scalers and encoders.
        
        Args:
            input_path (str): Path to load the pickle file.
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data.get('scalers', {})
            self.encoders = data.get('encoders', {})
        logger.info(f"Scalers and encoders loaded from {input_path}")


def feature_engineering_pipeline(config_dict: Dict, df: pd.DataFrame, 
                                 drop_original_features: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    This function orchestrates the entire feature engineering workflow:
    1. Set data
    2. Normalize features
    3. Standardize features
    4. Label encode categorical features
    5. One-hot encode categorical features
    6. Extract new features (CustomerCategory, SatisfactionCategory)
    7. Drop irrelevant features
    
    Args:
        config_dict (Dict): Configuration dictionary.
        df (pd.DataFrame): Input dataframe.
        drop_original_features (bool): Whether to drop original features. Default is True.
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe.
    """
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Initialize feature engineer
    engineer = FeatureEngineer(config=config_dict)
    engineer.set_data(df)

    # Feature transformation
    normalization_features = config_dict.get('feature_transformation', {}).get('normalization', [])
    if normalization_features:
        engineer.normalize_features(normalization_features, column_suffix="_norm")

    standardization_features = config_dict.get('feature_transformation', {}).get('standardization', [])
    if standardization_features:
        engineer.standardize_features(standardization_features, column_suffix="_std")

    # Feature encoding
    encoding_config = config_dict.get('encoding', {})
    
    # Label encoding
    if encoding_config.get('label_encoding'):
        # For Gender, create mapping
        label_mapping = {
            'Gender': {'Female': 0, 'Male': 1}
        }
        engineer.label_encode_features(label_mapping)

    # One-hot encoding
    if encoding_config.get('onehot_encoding'):
        engineer.onehot_encode_features(encoding_config.get('onehot_encoding', []))

    # Feature extraction
    engineer.extract_customer_category("OrderCount")
    engineer.extract_satisfaction_category("SatisfactionScore")

    # Feature selection - Drop irrelevant features
    drop_features = config_dict.get('features', {}).get('drop_features', [])
    if drop_features and drop_original_features:
        engineer.drop_features(drop_features)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETED")
    logger.info("=" * 60)

    return engineer.get_engineered_data()
