"""Audit persisted training artifacts for leakage and metadata consistency."""

import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from config import (
    ARTIFACT_SCHEMA_VERSION,
    DATA_PATH,
    EVENT_TIMESTAMP_COLUMN,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NON_MODEL_INPUT_FEATURES,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    RAW_FEATURES,
)
from src.bootstrap import model_artifacts_are_valid
from src.evaluation import evaluate_classifier
from src.train_model import split_training_dataset
from src.utils import read_json, write_json


def _timestamped_training_frame() -> pd.DataFrame:
    """Build an out-of-order frame whose latest two rows contain both classes."""
    row_count = 10
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{index}" for index in range(row_count)],
            "user_id": [f"USR{index % 3}" for index in range(row_count)],
            "merchant_id": [f"MER{index % 2}" for index in range(row_count)],
            EVENT_TIMESTAMP_COLUMN: pd.date_range(
                "2024-01-01", periods=row_count, freq="h", tz="UTC"
            )[::-1],
            "amount": np.arange(1, row_count + 1, dtype=float) * 100.0,
            "previous_failed_txns": np.zeros(row_count, dtype=int),
            "txn_count_10min": np.ones(row_count, dtype=int),
            "avg_user_transaction_amount": np.full(row_count, 100.0),
            "location_change": np.zeros(row_count, dtype=int),
            "account_age_days": np.arange(100, 100 + row_count),
            "hour_of_day": np.arange(row_count),
            "is_weekend": np.zeros(row_count, dtype=int),
            "international_transaction": np.zeros(row_count, dtype=int),
            "fraud": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def _copy_artifact_set(tmp_path, monkeypatch):
    """Copy the real compatible artifact set and redirect bootstrap to it."""
    from src import bootstrap

    copied_model = tmp_path / "fraud_model.pkl"
    copied_preprocessor = tmp_path / "preprocessor.pkl"
    copied_metadata = tmp_path / "model_metadata.json"
    shutil.copy2(MODEL_PATH, copied_model)
    shutil.copy2(PREPROCESSOR_PATH, copied_preprocessor)
    shutil.copy2(MODEL_METADATA_PATH, copied_metadata)
    monkeypatch.setattr(bootstrap, "MODEL_PATH", copied_model)
    monkeypatch.setattr(bootstrap, "PREPROCESSOR_PATH", copied_preprocessor)
    monkeypatch.setattr(bootstrap, "MODEL_METADATA_PATH", copied_metadata)
    return bootstrap, copied_model, copied_metadata


def test_ids_and_target_never_enter_model_features() -> None:
    forbidden = {
        "transaction_id",
        "user_id",
        "merchant_id",
        EVENT_TIMESTAMP_COLUMN,
        "fraud",
    }
    assert forbidden.isdisjoint(MODEL_FEATURES)
    assert set(NON_MODEL_INPUT_FEATURES).isdisjoint(MODEL_FEATURES)
    assert set(MODEL_FEATURES).issubset(
        set(RAW_FEATURES)
        | {
            "amount_ratio",
            "is_high_amount",
            "is_high_velocity",
            "failed_attempt_risk",
            "unusual_hour",
        }
    )


def test_saved_preprocessor_was_fitted_on_training_partition() -> None:
    dataframe = pd.read_csv(DATA_PATH)
    _, x_train, _, _, _, _ = split_training_dataset(dataframe)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata = read_json(MODEL_METADATA_PATH)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    assert np.allclose(scaler.mean_, x_train[NUMERICAL_FEATURES].mean().to_numpy())
    assert list(preprocessor.feature_names_in_) == MODEL_FEATURES
    assert metadata["tuning_rows"] <= metadata["refit_rows"]
    assert metadata["refit_rows"] == metadata["training_rows"] == len(x_train)


def test_saved_metrics_recompute_from_held_out_partition() -> None:
    dataframe = pd.read_csv(DATA_PATH)
    _, _, x_test, _, y_test, _ = split_training_dataset(dataframe)
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


def test_model_selection_uses_training_only_cross_validation() -> None:
    metadata = read_json(MODEL_METADATA_PATH)
    assert metadata["model_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["selection_partition"] == "training_only_cross_validation"
    selected_by_cv = max(
        metadata["models"],
        key=lambda name: (
            metadata["models"][name]["cv_selection_score"],
            metadata["models"][name]["cv_metrics"]["roc_auc"],
            metadata["models"][name]["cv_metrics"]["precision"],
        ),
    )
    assert metadata["selected_model"] == selected_by_cv
    for metrics in metadata["models"].values():
        assert (
            metrics["cv_metrics"]["false_positive_rate"]
            <= metadata["maximum_cv_false_positive_rate"]
        )
        assert metrics["best_parameters"]


def test_timestamped_dataset_uses_locked_chronological_split() -> None:
    dataframe = _timestamped_training_frame()
    _, x_train, x_test, y_train, y_test, split_metadata = split_training_dataset(
        dataframe
    )

    assert split_metadata["split_strategy"] == "chronological_80_20"
    assert split_metadata["training_period_end"] <= split_metadata["test_period_start"]
    assert len(x_train) == len(y_train) == 8
    assert len(x_test) == len(y_test) == 2
    assert x_train.index.tolist() == list(range(9, 1, -1))
    assert x_test.index.tolist() == [1, 0]
    assert set(y_train) == set(y_test) == {0, 1}
    assert {
        "transaction_id",
        "user_id",
        "merchant_id",
        EVENT_TIMESTAMP_COLUMN,
        "fraud",
    }.isdisjoint(x_train.columns)


def test_chronological_split_rejects_invalid_timestamp() -> None:
    dataframe = _timestamped_training_frame()
    dataframe[EVENT_TIMESTAMP_COLUMN] = dataframe[EVENT_TIMESTAMP_COLUMN].astype(
        "object"
    )
    dataframe.loc[4, EVENT_TIMESTAMP_COLUMN] = "not-a-timestamp"

    with pytest.raises(ValueError, match="valid timestamps"):
        split_training_dataset(dataframe)


def test_chronological_split_requires_both_classes_in_test() -> None:
    dataframe = _timestamped_training_frame()
    dataframe.loc[[0, 1], "fraud"] = 0

    with pytest.raises(ValueError, match="test partition"):
        split_training_dataset(dataframe)


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


def test_stale_artifact_feature_contract_is_detected(tmp_path, monkeypatch) -> None:
    bootstrap, _, metadata_path = _copy_artifact_set(tmp_path, monkeypatch)
    metadata = read_json(metadata_path)
    metadata["feature_list"] = list(reversed(metadata["feature_list"]))
    write_json(metadata_path, metadata)

    assert not bootstrap.model_artifacts_are_valid()


def test_model_preprocessor_dimension_mismatch_is_detected(
    tmp_path, monkeypatch
) -> None:
    bootstrap, model_path, _ = _copy_artifact_set(tmp_path, monkeypatch)
    model = joblib.load(model_path)
    model.n_features_in_ = int(model.n_features_in_) + 1
    joblib.dump(model, model_path)

    assert not bootstrap.model_artifacts_are_valid()
