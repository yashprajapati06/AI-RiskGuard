"""Tests for combined score bounds and risk thresholds."""

import pytest

from src.risk_engine import calculate_final_risk, classify_risk, combine_risk_scores


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "LOW"),
        (10, "LOW"),
        (34.999, "LOW"),
        (35, "MEDIUM"),
        (50, "MEDIUM"),
        (69.999, "MEDIUM"),
        (70, "HIGH"),
        (90, "HIGH"),
        (100, "HIGH"),
    ],
)
def test_risk_classification(score: float, expected: str) -> None:
    assert classify_risk(score) == expected


@pytest.mark.parametrize(
    ("ml_score", "rule_score", "expected"),
    [(-50, -10, 0), (150, 120, 100), (80, 60, 74)],
)
def test_final_score_is_bounded_and_weighted(
    ml_score: float, rule_score: float, expected: float
) -> None:
    assert combine_risk_scores(ml_score, rule_score) == expected


def test_action_matches_level() -> None:
    result = calculate_final_risk(80, 60)
    assert result["risk_level"] == "HIGH"
    assert result["recommended_action"] == "Flag for Immediate Manual Review"


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_risk_score_is_rejected(invalid_score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        combine_risk_scores(invalid_score, 50)
