"""Transaction scoring service."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import joblib

from config import MODEL_FEATURES, MODEL_PATH, PREPROCESSOR_PATH
from src.evaluation import predict_fraud_probabilities
from src.feature_engineering import engineer_features
from src.risk_engine import calculate_final_risk
from src.rule_engine import evaluate_rules
from src.utils import clamp
from src.validation import validate_transaction

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model_artifacts() -> tuple[Any, Any]:
    """Load the model files, retraining if they are missing or stale."""
    from src.bootstrap import model_artifacts_are_valid

    if not model_artifacts_are_valid():
        LOGGER.warning(
            "Model artifacts are missing or incompatible; starting training."
        )
        from src.train_model import train_and_save_models

        train_and_save_models()
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as exc:
        LOGGER.exception("Failed to load model artifacts")
        raise RuntimeError(
            "Model artifacts could not be loaded. Run: python -m src.train_model"
        ) from exc
    LOGGER.info("Loaded fraud model artifacts")
    return model, preprocessor


def clear_model_cache() -> None:
    """Clear the model cache after retraining."""
    load_model_artifacts.cache_clear()


def analyze_transaction(transaction_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and score one transaction without writing it to the database."""
    clean_transaction = validate_transaction(transaction_data)
    engineered = engineer_features(clean_transaction)
    model, preprocessor = load_model_artifacts()

    try:
        transformed = preprocessor.transform(engineered[MODEL_FEATURES])
        fraud_probability = clamp(
            float(predict_fraud_probabilities(model, transformed)[0]), 0.0, 1.0
        )
        model_prediction = int(model.predict(transformed)[0])
    except (ValueError, TypeError, AttributeError) as exc:
        LOGGER.exception("Prediction failed after transaction validation")
        raise RuntimeError(
            "The transaction could not be analyzed by the model."
        ) from exc

    ml_risk_score = clamp(fraud_probability * 100.0)
    rule_result = evaluate_rules(clean_transaction)
    final_result = calculate_final_risk(
        ml_risk_score=ml_risk_score,
        rule_risk_score=rule_result["rule_risk_score"],
    )
    return {
        "fraud_probability": round(fraud_probability, 6),
        "ml_risk_score": round(ml_risk_score, 2),
        "rule_risk_score": rule_result["rule_risk_score"],
        "final_risk_score": final_result["final_risk_score"],
        "risk_level": final_result["risk_level"],
        "recommended_action": final_result["recommended_action"],
        "triggered_rules": rule_result["triggered_rules"],
        "risk_reasons": rule_result["reasons"],
        "model_prediction": model_prediction,
    }
