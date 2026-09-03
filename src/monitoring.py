"""Monitoring data for predictions and model performance."""

from __future__ import annotations

from typing import Any

from config import MODEL_METADATA_PATH
from src.database import get_dashboard_summary
from src.utils import read_json, validate_model_metadata


def get_monitoring_summary() -> dict[str, Any]:
    """Combine saved prediction totals with test-set metrics."""
    live = get_dashboard_summary()
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        try:
            metadata = read_json(MODEL_METADATA_PATH)
            validate_model_metadata(metadata)
        except (OSError, ValueError, TypeError):
            metadata = {}
    selected_name = metadata.get("selected_model", "Unavailable")
    evaluation = metadata.get("models", {}).get(selected_name, {})
    return {
        "average_prediction_risk": live["average_risk"],
        "risk_level_distribution": {
            "LOW": live["low_count"],
            "MEDIUM": live["medium_count"],
            "HIGH": live["high_count"],
        },
        "high_risk_predictions": live["high_count"],
        "false_positive_rate_on_test_set": evaluation.get("false_positive_rate"),
        "false_negative_rate_on_test_set": evaluation.get("false_negative_rate"),
        "training_timestamp": metadata.get("training_timestamp"),
        "model_name": selected_name,
        "model_version": metadata.get("model_version", "Unavailable"),
        "live_accuracy_notice": (
            "Live accuracy cannot be calculated until ground-truth labels are available."
        ),
    }
