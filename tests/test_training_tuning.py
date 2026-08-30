"""Tests for leakage-safe cross-validation candidate selection."""

import numpy as np
import pandas as pd
import pytest

from src.train_model import _select_cv_candidate, _training_tuning_sample


def test_cv_candidate_selection_enforces_false_positive_constraint() -> None:
    cv_results = {
        "params": [{"candidate": "too_many_alerts"}, {"candidate": "eligible"}],
        "mean_test_precision": np.array([0.20, 0.25]),
        "mean_test_recall": np.array([0.95, 0.70]),
        "mean_test_f1": np.array([0.33, 0.37]),
        "mean_test_roc_auc": np.array([0.85, 0.82]),
        "mean_test_specificity": np.array([0.60, 0.96]),
    }

    assert _select_cv_candidate(cv_results) == 1


def test_cv_candidate_selection_uses_composite_within_constraint() -> None:
    cv_results = {
        "params": [{"candidate": "weaker"}, {"candidate": "stronger"}],
        "mean_test_precision": np.array([0.20, 0.25]),
        "mean_test_recall": np.array([0.55, 0.65]),
        "mean_test_f1": np.array([0.29, 0.36]),
        "mean_test_roc_auc": np.array([0.76, 0.80]),
        "mean_test_specificity": np.array([0.97, 0.95]),
    }

    assert _select_cv_candidate(cv_results) == 1


def test_cv_candidate_selection_rejects_all_excessive_fpr_options() -> None:
    cv_results = {
        "params": [{"candidate": "first"}, {"candidate": "second"}],
        "mean_test_precision": np.array([0.20, 0.25]),
        "mean_test_recall": np.array([0.90, 0.85]),
        "mean_test_f1": np.array([0.33, 0.39]),
        "mean_test_roc_auc": np.array([0.85, 0.84]),
        "mean_test_specificity": np.array([0.70, 0.75]),
    }

    with pytest.raises(ValueError, match="false-positive-rate"):
        _select_cv_candidate(cv_results)


def test_tuning_sample_contains_training_rows_only(monkeypatch) -> None:
    from src import train_model

    training_indices = range(80)
    held_out_indices = set(range(80, 100))
    features = pd.DataFrame(
        {"amount": np.arange(80, dtype=float)}, index=training_indices
    )
    target = pd.Series([0, 1] * 40, index=training_indices)
    monkeypatch.setattr(train_model, "CV_TUNING_MAX_ROWS", 20)

    tuning_features, tuning_target = _training_tuning_sample(features, target)

    assert len(tuning_features) == len(tuning_target) == 20
    assert set(tuning_features.index) == set(tuning_target.index)
    assert set(tuning_features.index).issubset(set(training_indices))
    assert set(tuning_features.index).isdisjoint(held_out_indices)
    assert tuning_target.value_counts().to_dict() == {0: 10, 1: 10}
