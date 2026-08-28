"""Model evaluation and selection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MODEL_SELECTION_WEIGHTS


def get_fraud_class_index(model: Any) -> int:
    """Return the probability-column index for fraud label 1.

    Classifiers usually expose classes as ``[0, 1]``, but probability extraction
    must follow the label rather than silently relying on that ordering.
    """
    classes = np.asarray(getattr(model, "classes_", []))
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("Classifier must expose exactly one fraud class labelled 1.")
    return int(matches[0])


def predict_fraud_probabilities(model: Any, features: Any) -> np.ndarray:
    """Return validated class-1 probabilities regardless of class ordering."""
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    fraud_index = get_fraud_class_index(model)
    if probabilities.ndim != 2 or probabilities.shape[1] <= fraud_index:
        raise ValueError("Classifier returned a malformed probability matrix.")
    fraud_probabilities = probabilities[:, fraud_index]
    if not np.isfinite(fraud_probabilities).all():
        raise ValueError("Classifier returned a non-finite fraud probability.")
    if ((fraud_probabilities < 0) | (fraud_probabilities > 1)).any():
        raise ValueError("Classifier returned a fraud probability outside [0, 1].")
    return fraud_probabilities


def evaluate_classifier(model: Any, features: Any, target: Any) -> dict[str, Any]:
    """Calculate actual fraud-class metrics from predictions and probabilities."""
    predictions = model.predict(features)
    probabilities = predict_fraud_probabilities(model, features)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "confusion_matrix": matrix.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def model_selection_score(metrics: dict[str, Any]) -> float:
    """Weight recall/F1 most heavily, with ROC-AUC and precision tie-breakers."""
    return float(
        sum(metrics[name] * weight for name, weight in MODEL_SELECTION_WEIGHTS.items())
    )


def select_best_model(results: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select the model with the highest documented fraud-oriented score."""
    if not results:
        raise ValueError("No model evaluation results were supplied.")
    ranked = sorted(
        results.items(),
        key=lambda item: (
            model_selection_score(item[1]),
            item[1]["roc_auc"],
            item[1]["precision"],
        ),
        reverse=True,
    )
    model_name, metrics = ranked[0]
    return model_name, metrics


def extract_feature_importance(
    model: Any, feature_names: np.ndarray
) -> list[dict[str, float | str]]:
    """Extract correctly aligned global importance from supported estimators."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        return []
    if len(values) != len(feature_names):
        return []
    order = np.argsort(values)[::-1]
    return [
        {"feature": str(feature_names[index]), "importance": float(values[index])}
        for index in order
    ]
