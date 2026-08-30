"""Tests for JSON robustness helpers."""

from copy import deepcopy

import pytest

from config import ARTIFACT_SCHEMA_VERSION
from src.utils import parse_json_list, read_json, validate_model_metadata


def _valid_metadata() -> dict:
    """Return a compact, internally consistent metadata document."""
    metrics = {
        "accuracy": 5 / 7,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "roc_auc": 0.75,
        "false_positive_rate": 0.2,
        "false_negative_rate": 0.5,
        "confusion_matrix": [[4, 1], [1, 1]],
        "tn": 4,
        "fp": 1,
        "fn": 1,
        "tp": 1,
        "selection_score": 0.55,
        "cv_selection_score": 0.54,
        "cv_metrics": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "roc_auc": 0.7,
            "specificity": 0.9,
            "false_positive_rate": 0.1,
            "selection_score": 0.53,
        },
        "best_parameters": {"class_weight": "balanced"},
    }
    return {
        "model_version": ARTIFACT_SCHEMA_VERSION,
        "selected_model": "logistic_regression",
        "selection_reason": "Training-only cross-validation selected the model.",
        "training_timestamp": "2026-08-30T10:00:00+00:00",
        "dataset_size": 10,
        "dataset_source_id": "test_fixture",
        "dataset_source": "Synthetic test fixture",
        "dataset_source_url": "local:test_fixture",
        "data_origin": "synthetic_test_fixture",
        "source_dataset_rows": 10,
        "source_sampling_strategy": "complete_fixture",
        "amount_normalization": "None.",
        "training_rows": 3,
        "test_rows": 7,
        "tuning_rows": 2,
        "refit_rows": 3,
        "fraud_rate": 0.2,
        "models": {
            "logistic_regression": deepcopy(metrics),
            "random_forest": deepcopy(metrics),
        },
        "feature_list": ["amount"],
        "non_model_input_features": ["device_type"],
        "transformed_feature_count": 1,
        "cv_folds": 2,
        "cv_strategy": "stratified_shuffled_training_only_sample",
        "maximum_cv_false_positive_rate": 0.2,
        "selection_partition": "training_only_cross_validation",
        "split_strategy": "stratified_random_80_20",
    }


def test_read_json_rejects_non_object(tmp_path) -> None:
    path = tmp_path / "invalid_shape.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected a JSON object"):
        read_json(path)


def test_malformed_stored_json_list_is_safe() -> None:
    assert parse_json_list("not-json") == []
    assert parse_json_list('{"wrong": "shape"}') == []
    assert parse_json_list('["NEW_DEVICE"]') == ["NEW_DEVICE"]


def test_model_metadata_rejects_incomplete_metrics() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_model_metadata({})


def test_model_metadata_accepts_consistent_document() -> None:
    validate_model_metadata(_valid_metadata())


def test_model_metadata_rejects_stale_schema_version() -> None:
    metadata = _valid_metadata()
    metadata["model_version"] = "1.1.0"

    with pytest.raises(ValueError, match="incompatible"):
        validate_model_metadata(metadata)


def test_model_metadata_rejects_incomplete_refit() -> None:
    metadata = _valid_metadata()
    metadata["refit_rows"] = metadata["training_rows"] - 1

    with pytest.raises(ValueError, match="refit_rows"):
        validate_model_metadata(metadata)


def test_model_metadata_rejects_overlapping_feature_contracts() -> None:
    metadata = _valid_metadata()
    metadata["non_model_input_features"] = ["amount"]

    with pytest.raises(ValueError, match="disjoint"):
        validate_model_metadata(metadata)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[4, -1], [1, 1]], "non-negative integers"),
        ([[4, 1], [1, 2]], "sum to test_rows"),
    ],
)
def test_model_metadata_rejects_invalid_confusion_matrix(
    matrix: list[list[int]], message: str
) -> None:
    metadata = _valid_metadata()
    metadata["models"]["logistic_regression"]["confusion_matrix"] = matrix

    with pytest.raises(ValueError, match=message):
        validate_model_metadata(metadata)


def test_model_metadata_rejects_inconsistent_named_confusion_counts() -> None:
    metadata = _valid_metadata()
    metadata["models"]["random_forest"]["tp"] = 2

    with pytest.raises(ValueError, match="named confusion counts"):
        validate_model_metadata(metadata)


def test_model_metadata_rejects_non_integer_named_confusion_counts() -> None:
    metadata = _valid_metadata()
    metadata["models"]["random_forest"]["tp"] = 1.0

    with pytest.raises(ValueError, match="non-negative integers"):
        validate_model_metadata(metadata)


def test_model_metadata_rejects_invalid_chronological_periods() -> None:
    metadata = _valid_metadata()
    metadata.update(
        {
            "split_strategy": "chronological_80_20",
            "training_period_start": "2024-01-01T00:00:00+00:00",
            "training_period_end": "2024-01-03T00:00:00+00:00",
            "test_period_start": "2024-01-02T00:00:00+00:00",
            "test_period_end": "2024-01-04T00:00:00+00:00",
        }
    )

    with pytest.raises(ValueError, match="valid and ordered"):
        validate_model_metadata(metadata)
