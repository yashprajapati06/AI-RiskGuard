"""Regression tests for probability indexing and confusion-matrix metrics."""

import numpy as np
import pytest

from src.evaluation import (
    evaluate_classifier,
    get_fraud_class_index,
    predict_fraud_probabilities,
    select_best_model,
)


class ReverseClassModel:
    classes_ = np.array([1, 0])

    def predict_proba(self, features):
        return np.tile([0.8, 0.2], (len(features), 1))


class ControlledModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([0, 1, 0, 1])

    def predict_proba(self, features):
        return np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])


def test_fraud_probability_uses_class_label_not_fixed_column() -> None:
    model = ReverseClassModel()
    assert get_fraud_class_index(model) == 0
    probabilities = predict_fraud_probabilities(model, np.zeros((2, 1)))
    assert probabilities.tolist() == pytest.approx([0.8, 0.8])


def test_confusion_matrix_indexing_is_tn_fp_fn_tp() -> None:
    target = np.array([0, 0, 1, 1])
    metrics = evaluate_classifier(ControlledModel(), np.zeros((4, 1)), target)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5


def test_model_selection_prioritizes_documented_composite() -> None:
    results = {
        "recall_model": {"recall": 0.8, "f1": 0.6, "roc_auc": 0.7, "precision": 0.4},
        "accuracy_only_model": {
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.55,
            "precision": 0.0,
        },
    }
    selected, _ = select_best_model(results)
    assert selected == "recall_model"
