"""Tests for the reusable prediction service."""

from src.predictor import analyze_transaction

REQUIRED_KEYS = {
    "fraud_probability",
    "ml_risk_score",
    "rule_risk_score",
    "final_risk_score",
    "risk_level",
    "recommended_action",
    "triggered_rules",
    "risk_reasons",
    "model_prediction",
}


def test_prediction_returns_required_keys(normal_transaction: dict) -> None:
    result = analyze_transaction(normal_transaction)
    assert REQUIRED_KEYS.issubset(result)
    assert 0 <= result["fraud_probability"] <= 1
    assert 0 <= result["final_risk_score"] <= 100


def test_demonstration_risk_order(sample_transactions: dict) -> None:
    results = {
        name: analyze_transaction(transaction)
        for name, transaction in sample_transactions.items()
    }
    assert results["normal"]["final_risk_score"] < results["medium"]["final_risk_score"]
    assert results["medium"]["final_risk_score"] < results["high"]["final_risk_score"]
    assert results["normal"]["risk_level"] == "LOW"
    assert results["medium"]["risk_level"] == "MEDIUM"
    assert results["high"]["risk_level"] == "HIGH"
