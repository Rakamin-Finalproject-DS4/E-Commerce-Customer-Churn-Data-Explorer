import pandas as pd
from src.model_trainer import ModelTrainer


def test_model_trainer_training_and_evaluation():
    X = pd.DataFrame({
        'feature_a': list(range(20)),
        'feature_b': list(range(20, 40))
    })
    y = pd.Series([0] * 14 + [1] * 6)

    trainer = ModelTrainer(config={
        'random_state': 42,
        'train_test_split': {'test_size': 0.3, 'random_state': 42},
        'imbalance_handling': {'method': 'RandomOverSampler'},
        'models': {'random_forest': {'hyperparameters': {'n_estimators': 10, 'max_depth': 3}, 'random_state': 42}}
    })

    trainer.prepare_data(X, y, test_size=0.3, random_state=42)
    X_resampled, y_resampled = trainer.handle_class_imbalance(method='RandomOverSampler')

    assert len(X_resampled) == len(y_resampled)
    assert y_resampled.value_counts().iloc[0] == y_resampled.value_counts().iloc[1]

    model = trainer.train_random_forest(use_balanced_data=True)
    predictions = trainer.predict(use_best_model=False)
    results = trainer.evaluate_model()

    assert 'accuracy' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1_score' in results
    assert model is not None
    assert len(predictions) == len(trainer.X_test)
    importance = trainer.get_feature_importance(top_n=2)
    assert importance.shape[0] == 2
