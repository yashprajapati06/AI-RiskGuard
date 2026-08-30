"""Shared utility helpers."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np


def configure_logging() -> None:
    """Configure concise application logging once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Clamp a numeric value to an inclusive range."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("Numeric value must be finite.")
    if minimum > maximum:
        raise ValueError("minimum cannot be greater than maximum.")
    return float(max(minimum, min(maximum, numeric_value)))


class JsonEncoder(json.JSONEncoder):
    """Serialize common NumPy scalar and array types in metadata."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write formatted UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, cls=JsonEncoder),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path.name}.")
    return payload


def parse_json_list(value: Any) -> list[str]:
    """Safely parse a stored JSON list for presentation."""
    try:
        payload = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def validate_model_metadata(metadata: dict[str, Any]) -> None:
    """Validate the persisted metadata structure used by monitoring pages."""
    required_top_level = {
        "selected_model",
        "training_timestamp",
        "dataset_size",
        "fraud_rate",
        "models",
        "feature_list",
        "cv_folds",
        "maximum_cv_false_positive_rate",
        "selection_partition",
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
    required_models = {"logistic_regression", "random_forest"}
    if not required_models.issubset(metadata["models"]):
        raise ValueError("Model metadata must contain both candidate model results.")
    if metadata["selected_model"] not in metadata["models"]:
        raise ValueError("Selected model is absent from model evaluation results.")
    if not isinstance(metadata["feature_list"], list) or not metadata["feature_list"]:
        raise TypeError("Model metadata feature_list must be a non-empty list.")
    if int(metadata["dataset_size"]) <= 0:
        raise ValueError("Model metadata dataset_size must be positive.")
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
    }
    for model_name, metrics in metadata["models"].items():
        if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
            raise ValueError(f"Metrics are incomplete for {model_name}.")
        scalar_metrics = required_metrics.difference(
            {"confusion_matrix", "cv_metrics", "best_parameters"}
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
        if (
            float(cv_metrics["false_positive_rate"])
            > maximum_cv_fpr + 1e-12
        ):
            raise ValueError(f"{model_name} exceeds the cross-validation FPR limit.")
        if not isinstance(metrics["best_parameters"], dict) or not metrics[
            "best_parameters"
        ]:
            raise TypeError(f"{model_name}.best_parameters must be a non-empty object.")
