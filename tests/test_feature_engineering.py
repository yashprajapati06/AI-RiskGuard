"""Tests for shared derived-feature logic."""

import numpy as np

from src.feature_engineering import engineer_features


def test_amount_ratio_is_correct(normal_transaction: dict) -> None:
    normal_transaction["amount"] = 600
    normal_transaction["avg_user_transaction_amount"] = 200
    result = engineer_features(normal_transaction)
    assert result.loc[0, "amount_ratio"] == 3.0


def test_amount_ratio_division_by_zero_is_safe(normal_transaction: dict) -> None:
    normal_transaction["amount"] = 100
    normal_transaction["avg_user_transaction_amount"] = 0
    result = engineer_features(normal_transaction)
    assert np.isfinite(result.loc[0, "amount_ratio"])
    assert result.loc[0, "amount_ratio"] > 0


def test_derived_flags_are_generated(high_transaction: dict) -> None:
    result = engineer_features(high_transaction).iloc[0]
    assert result["is_high_amount"] == 1
    assert result["is_high_velocity"] == 1
    assert result["failed_attempt_risk"] == 1
    assert result["unusual_hour"] == 1
