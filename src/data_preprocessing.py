"""
Data Preprocessing Module for E-Commerce Customer Churn Prediction

This module handles all data cleaning, transformation, and preparation tasks
including missing value imputation, outlier detection/removal, and encoding.

Author: Data Science Team
Version: 1.0
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Tuple, List, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle data preprocessing tasks for customer churn prediction.
    
    This class provides methods for:
    - Loading raw data from Excel
    - Handling missing values
    - Detecting and removing outliers
    - Data cleansing (duplicates, inconsistent values)
    - Feature transformation
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the DataPreprocessor with configuration parameters.
        
        Args:
            config (Dict, optional): Configuration dictionary containing preprocessing parameters.
                                    Defaults to None.
        """
        self.config = config or {}
        self.df = None
        self.numerical_cols = None
        self.categorical_cols = None

    def load_data(self, file_path: str, sheet_name: str = "E Comm") -> pd.DataFrame:
        """
        Load raw data from Excel file.
        
        Args:
            file_path (str): Path to the Excel file.
            sheet_name (str): Name of the sheet to load. Defaults to "E Comm".
            
        Returns:
            pd.DataFrame: Loaded dataframe.
            
        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the sheet name doesn't exist.
        """
        try:
            logger.info(f"Loading data from {file_path}, sheet: {sheet_name}")
            self.df = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except ValueError as e:
            logger.error(f"Invalid sheet name: {sheet_name}")
            raise

    def set_feature_columns(self, numerical_cols: List[str], categorical_cols: List[str]):
        """
        Set the numerical and categorical feature columns.
        
        Args:
            numerical_cols (List[str]): List of numerical column names.
            categorical_cols (List[str]): List of categorical column names.
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        logger.info(f"Set {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")

    def check_duplicates(self) -> int:
        """
        Check for duplicate rows in the dataset.
        
        Returns:
            int: Number of duplicate rows found.
        """
        duplicates = self.df.duplicated().sum()
        logger.info(f"Found {duplicates} duplicate rows")
        return duplicates

    def remove_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        """
        initial_shape = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        final_shape = self.df.shape[0]
        logger.info(f"Removed {initial_shape - final_shape} duplicate rows")

    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the dataset.
        
        Returns:
            pd.Series: Count of missing values for each column (sorted descending).
        """
        missing = self.df.isna().sum().sort_values(ascending=False)
        missing_percent = (missing / len(self.df) * 100).round(2)
        logger.info("Missing values detected:")
        for col, count, percent in zip(missing.index, missing.values, missing_percent.values):
            if count > 0:
                logger.info(f"  {col}: {count} ({percent}%)")
        return missing

    def impute_missing_values(self, strategy: Dict = None) -> pd.DataFrame:
        """
        Impute missing values using median or mean based on strategy.
        
        Strategy:
            - 'median': Use median value (for skewed distributions)
            - 'mean': Use mean value (for normal distributions)
            
        Args:
            strategy (Dict, optional): Dictionary with column names as keys and 'median'/'mean' as values.
                                       If None, uses default strategy from config.
                                       
        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        if strategy is None:
            strategy = {}

        # Default strategy from config or simple defaults
        default_strategy = {
            'DaySinceLastOrder': 'median',
            'OrderAmountHikeFromlastYear': 'median',
            'Tenure': 'median',
            'OrderCount': 'median',
            'CouponUsed': 'median',
            'WarehouseToHome': 'median',
            'HourSpendOnApp': 'mean',
        }
        
        # Merge with provided strategy
        final_strategy = {**default_strategy, **strategy}

        logger.info("Starting imputation process...")
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                if self.df[col].dtype == object:
                    impute_value = self.df[col].mode()
                    impute_value = impute_value.iloc[0] if not impute_value.empty else ''
                    self.df[col] = self.df[col].fillna(impute_value)
                    logger.info(f"  {col}: Imputed categorical values with mode = {impute_value}")
                    continue

                impute_method = final_strategy.get(col, 'median')

                if impute_method == 'median':
                    impute_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(impute_value)
                    logger.info(f"  {col}: Imputed with median = {impute_value:.2f}")
                    
                elif impute_method == 'mean':
                    impute_value = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(impute_value)
                    logger.info(f"  {col}: Imputed with mean = {impute_value:.2f}")

        logger.info("Imputation completed")
        return self.df

    def remove_outliers_zscore(self, threshold: float = 3) -> Tuple[pd.DataFrame, int]:
        """
        Remove outliers using Z-score method.
        
        Any value with absolute Z-score > threshold is considered an outlier.
        
        Args:
            threshold (float): Z-score threshold. Default is 3.
            
        Returns:
            Tuple[pd.DataFrame, int]: Filtered dataframe and number of rows removed.
        """
        if self.numerical_cols is None:
            logger.warning("Numerical columns not set. Skipping outlier removal.")
            return self.df, 0

        initial_count = len(self.df)
        filtered_entries = np.array([True] * len(self.df))

        for col in self.numerical_cols:
            if col in self.df.columns:
                zscore = np.abs(stats.zscore(self.df[col]))
                filtered_entries = (zscore < threshold) & filtered_entries

        self.df = self.df[filtered_entries]
        removed_count = initial_count - len(self.df)
        
        logger.info(f"Removed {removed_count} outliers ({removed_count/initial_count*100:.2f}%) using Z-score method")
        return self.df, removed_count

    def remove_outliers_iqr(self, multiplier: float = 1.5) -> Tuple[pd.DataFrame, int]:
        """
        Remove outliers using Interquartile Range (IQR) method.
        
        Values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR are considered outliers.
        
        Args:
            multiplier (float): IQR multiplier. Default is 1.5.
            
        Returns:
            Tuple[pd.DataFrame, int]: Filtered dataframe and number of rows removed.
        """
        if self.numerical_cols is None:
            logger.warning("Numerical columns not set. Skipping outlier removal.")
            return self.df, 0

        initial_count = len(self.df)
        
        for col in self.numerical_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                low_limit = Q1 - (multiplier * IQR)
                high_limit = Q3 + (multiplier * IQR)
                self.df = self.df[(self.df[col] >= low_limit) & (self.df[col] <= high_limit)]

        removed_count = initial_count - len(self.df)
        logger.info(f"Removed {removed_count} outliers ({removed_count/initial_count*100:.2f}%) using IQR method")
        return self.df, removed_count

    def clean_categorical_values(self, replacements: Dict[str, Dict[str, str]] = None):
        """
        Clean and standardize categorical values.
        
        Handles cases like 'Phone' vs 'Mobile Phone', 'CC' vs 'Credit Card', etc.
        
        Args:
            replacements (Dict): Dictionary with column names as keys and replacement dicts as values.
                                 Example: {'PreferredLoginDevice': {'Phone': 'Mobile Phone'}}
        """
        if replacements is None:
            replacements = {}

        # Default replacements
        default_replacements = {
            'PreferredLoginDevice': {'Phone': 'Mobile Phone'},
            'PreferredPaymentMode': {
                'CC': 'Credit Card',
                'COD': 'Cash on Delivery'
            },
            'PreferedOrderCat': {'Mobile': 'Mobile Phone'}
        }

        # Merge with provided replacements
        final_replacements = {**default_replacements, **replacements}

        for col, repl_dict in final_replacements.items():
            if col in self.df.columns:
                for old_val, new_val in repl_dict.items():
                    count = (self.df[col] == old_val).sum()
                    if count > 0:
                        self.df[col] = self.df[col].replace(old_val, new_val)
                        logger.info(f"  {col}: Replaced '{old_val}' with '{new_val}' ({count} instances)")

        logger.info("Categorical value cleaning completed")

    def get_data_info(self) -> Dict:
        """
        Get comprehensive information about the current dataset.
        
        Returns:
            Dict: Dictionary containing dataset statistics.
        """
        info = {
            'shape': self.df.shape,
            'missing_values': self.df.isna().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'dtypes': self.df.dtypes.to_dict(),
            'numerical_summary': self.df[self.numerical_cols].describe().to_dict() if self.numerical_cols else None
        }
        return info

    def get_preprocessed_data(self) -> pd.DataFrame:
        """
        Get the preprocessed dataframe.
        
        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        return self.df

    def save_preprocessed_data(self, output_path: str):
        """
        Save the preprocessed dataframe to CSV.
        
        Args:
            output_path (str): Path to save the CSV file.
        """
        self.df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")


def preprocess_pipeline(config_dict: Dict, input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Complete preprocessing pipeline from raw data to clean data.
    
    This function orchestrates the entire preprocessing workflow:
    1. Load data
    2. Set feature columns
    3. Remove duplicates
    4. Impute missing values
    5. Remove outliers
    6. Clean categorical values
    
    Args:
        config_dict (Dict): Configuration dictionary.
        input_path (str): Path to raw data file.
        output_path (str, optional): Path to save preprocessed data.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(config=config_dict)

    # Load data
    preprocessor.load_data(input_path, sheet_name=config_dict.get('data', {}).get('raw_sheet', 'E Comm'))

    # Set feature columns
    preprocessor.set_feature_columns(
        config_dict.get('numerical_features', []),
        config_dict.get('categorical_features', [])
    )

    # Check initial state
    preprocessor.check_missing_values()
    preprocessor.check_duplicates()

    # Data cleaning
    preprocessor.remove_duplicates()
    preprocessor.impute_missing_values()
    preprocessor.clean_categorical_values()

    # Outlier removal
    preprocessor.remove_outliers_zscore(threshold=config_dict.get('outlier_handling', {}).get('zscore_threshold', 3))

    # Save if output path provided
    if output_path:
        preprocessor.save_preprocessed_data(output_path)

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETED")
    logger.info("=" * 60)

    return preprocessor.get_preprocessed_data()
