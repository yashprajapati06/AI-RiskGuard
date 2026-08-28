"""Tests for transparent prototype rule scoring."""

import pytest

from src.rule_engine import evaluate_rules
from src.validation import TransactionValidationError


def test_high_risk_combination_scores_higher(
    normal_transaction: dict, high_transaction: dict
) -> None:
    normal_result = evaluate_rules(normal_transaction)
    high_result = evaluate_rules(high_transaction)
    assert high_result["rule_risk_score"] > normal_result["rule_risk_score"]
    assert high_result["rule_risk_score"] >= 80
    assert "HIGH_VELOCITY" in high_result["triggered_rules"]
    assert len(high_result["reasons"]) == len(high_result["triggered_rules"])


def test_rule_score_is_bounded(high_transaction: dict) -> None:
    result = evaluate_rules(high_transaction)
    assert 0 <= result["rule_risk_score"] <= 100


def test_rule_engine_rejects_invalid_transaction(normal_transaction: dict) -> None:
    normal_transaction["merchant_risk_score"] = 1.5
    with pytest.raises(TransactionValidationError):
        evaluate_rules(normal_transaction)
