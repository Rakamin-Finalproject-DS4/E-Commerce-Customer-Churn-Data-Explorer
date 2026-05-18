import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scripts.save_artifacts import ArtifactManager


def test_artifact_manager_saves_and_loads(tmp_path):
    artifact_dir = tmp_path / 'models'
    manager = ArtifactManager(str(artifact_dir))

    dummy_model = {'test': 123}
    model_path = manager.save_model(dummy_model, model_name='dummy_model')
    assert Path(model_path).exists()

    scaler = StandardScaler()
    scaler.fit([[0], [1], [2]])
    scaler_path = manager.save_scaler(scaler, scaler_name='dummy_scaler')
    assert Path(scaler_path).exists()

    features_path = manager.save_feature_list(['feature_a', 'feature_b'], feature_list_name='dummy_features')
    assert Path(features_path).exists()

    metadata_path = manager.save_model_metadata({'accuracy': 0.95}, metadata_name='dummy_metadata')
    assert Path(metadata_path).exists()

    loaded_model = manager.load_model(model_path)
    assert loaded_model == dummy_model

    loaded_scaler = manager.load_scaler(scaler_path)
    assert loaded_scaler is not None

    loaded_features = manager.load_feature_list(features_path)
    assert loaded_features == ['feature_a', 'feature_b']

    loaded_metadata = manager.load_metadata(metadata_path)
    assert loaded_metadata['accuracy'] == 0.95

    artifacts = manager.list_artifacts()
    assert 'dummy_model.pkl' in artifacts
    assert 'dummy_scaler.pkl' in artifacts
    assert 'dummy_features.json' in artifacts
    assert 'dummy_metadata.json' in artifacts
