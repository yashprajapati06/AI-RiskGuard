"""Combine machine-learning and rule scores into a bounded recommendation."""

from __future__ import annotations

from typing import Any

from config import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    ML_WEIGHT,
    RISK_ACTIONS,
    RULE_WEIGHT,
)
from src.utils import clamp


def classify_risk(score: float) -> str:
    """Classify a bounded score with configurable LOW/MEDIUM/HIGH thresholds."""
    bounded = clamp(float(score))
    if bounded < LOW_RISK_THRESHOLD:
        return "LOW"
    if bounded < HIGH_RISK_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def calculate_final_risk(
    ml_risk_score: float,
    rule_risk_score: float,
    ml_weight: float = ML_WEIGHT,
    rule_weight: float = RULE_WEIGHT,
) -> dict[str, Any]:
    """Return a normalized weighted score, level, and prototype action."""
    if ml_weight < 0 or rule_weight < 0 or ml_weight + rule_weight <= 0:
        raise ValueError("Risk weights must be non-negative and have a positive total.")
    normalized_ml_weight = ml_weight / (ml_weight + rule_weight)
    normalized_rule_weight = rule_weight / (ml_weight + rule_weight)
    score = clamp(
        clamp(float(ml_risk_score)) * normalized_ml_weight
        + clamp(float(rule_risk_score)) * normalized_rule_weight
    )
    rounded_score = round(score, 2)
    level = classify_risk(rounded_score)
    return {
        "final_risk_score": rounded_score,
        "risk_score": rounded_score,
        "risk_level": level,
        "recommended_action": RISK_ACTIONS[level],
    }


def combine_risk_scores(ml_risk_score: float, rule_risk_score: float) -> float:
    """Convenience API returning only the combined bounded score."""
    return calculate_final_risk(ml_risk_score, rule_risk_score)["final_risk_score"]
