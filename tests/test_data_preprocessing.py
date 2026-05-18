import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor


def test_impute_missing_values_and_clean_categorical_values():
    df = pd.DataFrame({
        'Tenure': [1.0, np.nan, 5.0],
        'HourSpendOnApp': [12.0, 20.0, np.nan],
        'CouponUsed': [1.0, np.nan, 2.0],
        'PreferredLoginDevice': ['Phone', None, 'Desktop'],
        'PreferredPaymentMode': ['CC', 'COD', None],
        'PreferedOrderCat': ['Mobile', 'Electronics', None]
    })

    preprocessor = DataPreprocessor()
    preprocessor.df = df.copy()
    preprocessor.set_feature_columns(
        numerical_cols=['Tenure', 'HourSpendOnApp', 'CouponUsed'],
        categorical_cols=['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat']
    )

    preprocessor.clean_categorical_values()
    result_df = preprocessor.impute_missing_values()

    assert result_df['PreferredLoginDevice'].iloc[0] == 'Mobile Phone'
    assert result_df['PreferredPaymentMode'].iloc[0] == 'Credit Card'
    assert result_df['PreferedOrderCat'].iloc[0] == 'Mobile Phone'
    assert result_df.isna().sum().sum() == 0


def test_remove_outliers_zscore():
    df = pd.DataFrame({'Tenure': [1.0, 2.0, 3.0, 50.0, 4.0]})
    preprocessor = DataPreprocessor()
    preprocessor.df = df.copy()
    preprocessor.set_feature_columns(numerical_cols=['Tenure'], categorical_cols=[])

    filtered_df, removed_count = preprocessor.remove_outliers_zscore(threshold=1.5)

    assert removed_count == 1
    assert 50.0 not in filtered_df['Tenure'].values
