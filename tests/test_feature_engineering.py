import pandas as pd
from src.feature_engineering import FeatureEngineer


def test_feature_engineering_workflow():
    df = pd.DataFrame({
        'OrderCount': [1, 5, 10],
        'SatisfactionScore': [1.2, 3.8, 4.9],
        'Gender': ['Female', 'Male', 'Female'],
        'MaritalStatus': ['Single', 'Married', 'Divorced'],
        'Tenure': [2.0, 10.0, 18.0],
        'CashbackAmount': [20.0, 34.0, 12.0]
    })

    engineer = FeatureEngineer()
    engineer.set_data(df)
    engineer.label_encode_features({'Gender': {'Female': 0, 'Male': 1}})
    engineer.onehot_encode_features(['MaritalStatus'])
    engineer.extract_customer_category(order_count_col='OrderCount')
    engineer.extract_satisfaction_category(satisfaction_col='SatisfactionScore')
    engineer.normalize_features(['CashbackAmount'], column_suffix='_norm')
    engineer.standardize_features(['Tenure'], column_suffix='_std')
    result_df = engineer.drop_features(['MaritalStatus'])

    assert 'Gender' in result_df.columns
    assert result_df['Gender'].tolist() == [0, 1, 0]
    assert 'MaritalStatus_Single' in result_df.columns
    assert 'CustomerCategory' in result_df.columns
    assert 'SatisfactionCategory' in result_df.columns
    assert 'CashbackAmount_norm' in result_df.columns
    assert 'Tenure_std' in result_df.columns
