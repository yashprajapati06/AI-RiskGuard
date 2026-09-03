"""Small helpers used across the project."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from config import ARTIFACT_SCHEMA_VERSION


def configure_logging() -> None:
    """Set the default application log format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Keep a number inside an inclusive range."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("Numeric value must be finite.")
    if minimum > maximum:
        raise ValueError("minimum cannot be greater than maximum.")
    return float(max(minimum, min(maximum, numeric_value)))


class JsonEncoder(json.JSONEncoder):
    """Handle NumPy values when writing metadata."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, cls=JsonEncoder),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path.name}.")
    return payload


def parse_json_list(value: Any) -> list[str]:
    """Parse a stored JSON list, returning an empty list on bad input."""
    try:
        payload = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def file_sha256(path: Path) -> str:
    """Hash a file in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_joblib_dump(payload: Any, path: Path, *, compress: int = 3) -> None:
    """Write a joblib artifact through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        joblib.dump(payload, temporary_path, compress=compress)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_model_metadata(metadata: dict[str, Any]) -> None:
    """Validate metadata used by training and monitoring."""
    required_top_level = {
        "model_version",
        "selected_model",
        "selection_reason",
        "training_timestamp",
        "dataset_size",
        "dataset_source_id",
        "dataset_source",
        "dataset_source_url",
        "data_origin",
        "source_dataset_rows",
        "source_sampling_strategy",
        "fraud_rate",
        "models",
        "feature_list",
        "non_model_input_features",
        "cv_folds",
        "cv_strategy",
        "maximum_cv_false_positive_rate",
        "selection_partition",
        "split_strategy",
        "training_rows",
        "test_rows",
        "tuning_rows",
        "refit_rows",
        "transformed_feature_count",
        "amount_normalization",
        "dataset_sha256",
        "model_sha256",
        "preprocessor_sha256",
    }
    missing = required_top_level.difference(metadata)
    if missing:
        raise ValueError("Model metadata is missing: " + ", ".join(sorted(missing)))
    if not isinstance(metadata["models"], dict):
        raise TypeError("Model metadata 'models' must be an object.")
    if (
        not isinstance(metadata["selected_model"], str)
        or not metadata["selected_model"]
    ):
        raise TypeError("Model metadata selected_model must be a non-empty string.")
    if (
        not isinstance(metadata["training_timestamp"], str)
        or not metadata["training_timestamp"]
    ):
        raise TypeError("Model metadata training_timestamp must be a non-empty string.")
    if metadata["model_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Model metadata version is incompatible with this application."
        )
    if (
        not isinstance(metadata["selection_reason"], str)
        or not metadata["selection_reason"].strip()
    ):
        raise TypeError("Model metadata selection_reason must be a non-empty string.")
    required_models = {"logistic_regression", "random_forest"}
    if not required_models.issubset(metadata["models"]):
        raise ValueError("Model metadata must contain both candidate model results.")
    if metadata["selected_model"] not in metadata["models"]:
        raise ValueError("Selected model is absent from model evaluation results.")
    if not isinstance(metadata["feature_list"], list) or not metadata["feature_list"]:
        raise TypeError("Model metadata feature_list must be a non-empty list.")
    if not isinstance(metadata["non_model_input_features"], list):
        raise TypeError("Model metadata non_model_input_features must be a list.")
    for key in ("feature_list", "non_model_input_features"):
        values = metadata[key]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise TypeError(f"Model metadata {key} must contain non-empty strings.")
        if len(values) != len(set(values)):
            raise ValueError(f"Model metadata {key} must not contain duplicates.")
    overlap = set(metadata["feature_list"]).intersection(
        metadata["non_model_input_features"]
    )
    if overlap:
        raise ValueError("Model and non-model feature lists must be disjoint.")
    for key in (
        "dataset_source_id",
        "dataset_source",
        "dataset_source_url",
        "data_origin",
        "source_sampling_strategy",
        "amount_normalization",
    ):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise TypeError(f"Model metadata {key} must be a non-empty string.")
    for key in ("dataset_sha256", "model_sha256", "preprocessor_sha256"):
        value = metadata[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError(f"Model metadata {key} must be a SHA-256 digest.")
    if int(metadata["dataset_size"]) <= 0:
        raise ValueError("Model metadata dataset_size must be positive.")
    dataset_size = int(metadata["dataset_size"])
    training_rows = int(metadata["training_rows"])
    test_rows = int(metadata["test_rows"])
    tuning_rows = int(metadata["tuning_rows"])
    refit_rows = int(metadata["refit_rows"])
    source_dataset_rows = int(metadata["source_dataset_rows"])
    transformed_feature_count = int(metadata["transformed_feature_count"])
    if training_rows + test_rows != dataset_size:
        raise ValueError("Training and test row counts must equal dataset_size.")
    if training_rows <= 0 or test_rows <= 0:
        raise ValueError("Training and test row counts must both be positive.")
    if not 0 < tuning_rows <= training_rows:
        raise ValueError(
            "tuning_rows must be positive and no larger than training_rows."
        )
    if refit_rows != training_rows:
        raise ValueError("refit_rows must equal training_rows.")
    if source_dataset_rows < dataset_size:
        raise ValueError("source_dataset_rows cannot be smaller than dataset_size.")
    if transformed_feature_count <= 0:
        raise ValueError("transformed_feature_count must be positive.")
    fraud_rate = float(metadata["fraud_rate"])
    if not math.isfinite(fraud_rate) or not 0 <= fraud_rate <= 1:
        raise ValueError("Model metadata fraud_rate must be between 0 and 1.")
    if int(metadata["cv_folds"]) < 2:
        raise ValueError("Model metadata cv_folds must be at least 2.")
    maximum_cv_fpr = float(metadata["maximum_cv_false_positive_rate"])
    if not math.isfinite(maximum_cv_fpr) or not 0 <= maximum_cv_fpr <= 1:
        raise ValueError(
            "Model metadata maximum_cv_false_positive_rate must be between 0 and 1."
        )
    if metadata["selection_partition"] != "training_only_cross_validation":
        raise ValueError("Model selection must use training-only cross-validation.")
    if metadata["cv_strategy"] not in {
        "stratified_shuffled_training_only_sample",
        "time_series_expanding_training_only_sample",
    }:
        raise ValueError("Model metadata contains an unsupported CV strategy.")
    if metadata["split_strategy"] not in {
        "chronological_80_20",
        "stratified_random_80_20",
    }:
        raise ValueError("Model metadata contains an unsupported split strategy.")
    if metadata["split_strategy"] == "chronological_80_20":
        period_keys = (
            "training_period_start",
            "training_period_end",
            "test_period_start",
            "test_period_end",
        )
        missing_periods = [key for key in period_keys if key not in metadata]
        if missing_periods:
            raise ValueError(
                "Chronological metadata is missing: " + ", ".join(missing_periods)
            )
        try:
            periods = [
                datetime.fromisoformat(str(metadata[key]).replace("Z", "+00:00"))
                for key in period_keys
            ]
            periods_are_ordered = all(
                earlier <= later for earlier, later in pairwise(periods)
            )
        except (TypeError, ValueError):
            periods_are_ordered = False
        if not periods_are_ordered:
            raise ValueError(
                "Chronological metadata periods must be valid and ordered."
            )

    required_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "confusion_matrix",
        "selection_score",
        "cv_selection_score",
        "cv_metrics",
        "best_parameters",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    for model_name, metrics in metadata["models"].items():
        if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
            raise ValueError(f"Metrics are incomplete for {model_name}.")
        scalar_metrics = required_metrics.difference(
            {
                "confusion_matrix",
                "cv_metrics",
                "best_parameters",
                "tn",
                "fp",
                "fn",
                "tp",
            }
        )
        for metric_name in scalar_metrics:
            value = float(metrics[metric_name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{model_name}.{metric_name} must be between 0 and 1.")
        matrix = metrics["confusion_matrix"]
        if (
            not isinstance(matrix, list)
            or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        ):
            raise ValueError(f"{model_name}.confusion_matrix must be a 2x2 list.")
        flat_matrix = [value for row in matrix for value in row]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in flat_matrix
        ):
            raise ValueError(
                f"{model_name}.confusion_matrix must contain non-negative integers."
            )
        if sum(flat_matrix) != test_rows:
            raise ValueError(f"{model_name}.confusion_matrix must sum to test_rows.")
        tn, fp, fn, tp = flat_matrix
        expected_counts = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        if any(
            isinstance(metrics[key], bool)
            or not isinstance(metrics[key], int)
            or metrics[key] < 0
            for key in expected_counts
        ):
            raise ValueError(
                f"{model_name} named confusion counts must be non-negative integers."
            )
        if any(metrics[key] != value for key, value in expected_counts.items()):
            raise ValueError(
                f"{model_name} named confusion counts do not match the matrix."
            )
        expected_fpr = fp / (fp + tn) if fp + tn else 0.0
        expected_fnr = fn / (fn + tp) if fn + tp else 0.0
        if not math.isclose(
            float(metrics["false_positive_rate"]),
            expected_fpr,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{model_name}.false_positive_rate is inconsistent with the matrix."
            )
        if not math.isclose(
            float(metrics["false_negative_rate"]),
            expected_fnr,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{model_name}.false_negative_rate is inconsistent with the matrix."
            )
        cv_metrics = metrics["cv_metrics"]
        if not isinstance(cv_metrics, dict):
            raise TypeError(f"{model_name}.cv_metrics must be an object.")
        required_cv_metrics = {
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "specificity",
            "false_positive_rate",
            "selection_score",
        }
        if not required_cv_metrics.issubset(cv_metrics):
            raise ValueError(f"CV metrics are incomplete for {model_name}.")
        for metric_name in required_cv_metrics:
            value = float(cv_metrics[metric_name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"{model_name}.cv_metrics.{metric_name} must be between 0 and 1."
                )
        if float(cv_metrics["false_positive_rate"]) > maximum_cv_fpr + 1e-12:
            raise ValueError(f"{model_name} exceeds the cross-validation FPR limit.")
        if (
            not isinstance(metrics["best_parameters"], dict)
            or not metrics["best_parameters"]
        ):
            raise TypeError(f"{model_name}.best_parameters must be a non-empty object.")
