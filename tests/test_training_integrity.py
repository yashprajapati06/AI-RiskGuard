"""Audit persisted training artifacts for leakage and metadata consistency."""

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.bootstrap import model_artifacts_are_valid
from src.evaluation import evaluate_classifier
from src.feature_engineering import engineer_features
from src.utils import read_json


def test_ids_and_target_never_enter_model_features() -> None:
    forbidden = {"transaction_id", "user_id", "merchant_id", "fraud"}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_saved_preprocessor_was_fitted_on_training_partition() -> None:
    import pandas as pd

    dataframe = engineer_features(pd.read_csv(DATA_PATH))
    features = dataframe[MODEL_FEATURES]
    target = dataframe["fraud"].astype(int)
    x_train, _, _, _ = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    assert np.allclose(scaler.mean_, x_train[NUMERICAL_FEATURES].mean().to_numpy())
    assert list(preprocessor.feature_names_in_) == MODEL_FEATURES


def test_saved_metrics_recompute_from_held_out_partition() -> None:
    import pandas as pd

    dataframe = engineer_features(pd.read_csv(DATA_PATH))
    features = dataframe[MODEL_FEATURES]
    target = dataframe["fraud"].astype(int)
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    recomputed = evaluate_classifier(model, preprocessor.transform(x_test), y_test)
    metadata = read_json(MODEL_METADATA_PATH)
    saved = metadata["models"][metadata["selected_model"]]
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
    ):
        assert recomputed[metric] == saved[metric]
    assert recomputed["confusion_matrix"] == saved["confusion_matrix"]


def test_complete_model_artifact_set_loads() -> None:
    assert model_artifacts_are_valid()


def test_incomplete_artifact_set_is_detected(tmp_path, monkeypatch) -> None:
    from src import bootstrap

    monkeypatch.setattr(bootstrap, "MODEL_PATH", tmp_path / "missing_model.pkl")
    monkeypatch.setattr(
        bootstrap, "PREPROCESSOR_PATH", tmp_path / "missing_preprocessor.pkl"
    )
    monkeypatch.setattr(
        bootstrap, "MODEL_METADATA_PATH", tmp_path / "missing_metadata.json"
    )
    assert not bootstrap.model_artifacts_are_valid()
