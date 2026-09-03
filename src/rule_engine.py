"""Simple risk rules used alongside the model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from config import RULE_THRESHOLDS, RULE_WEIGHTS, UNUSUAL_HOURS
from src.feature_engineering import engineer_features
from src.utils import clamp
from src.validation import validate_transaction


def evaluate_rules(transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Run the rule checks and return their combined score."""
    enriched = engineer_features(validate_transaction(transaction)).iloc[0].to_dict()
    triggered_rules: list[str] = []
    reasons: list[str] = []
    raw_score = 0.0

    def trigger(code: str, reason: str) -> None:
        nonlocal raw_score
        triggered_rules.append(code)
        reasons.append(reason)
        raw_score += RULE_WEIGHTS[code]

    amount = float(enriched["amount"])
    amount_ratio = float(enriched["amount_ratio"])
    if amount >= RULE_THRESHOLDS["high_amount"]:
        trigger(
            "VERY_HIGH_AMOUNT",
            f"High-value transaction detected (₹{amount:,.2f}).",
        )
    if int(enriched["is_new_device"]) == 1:
        trigger("NEW_DEVICE", "Transaction originated from a new device.")
    if int(enriched["previous_failed_txns"]) >= RULE_THRESHOLDS["failed_transactions"]:
        trigger(
            "MULTIPLE_FAILED_ATTEMPTS",
            f"{int(enriched['previous_failed_txns'])} recent failed attempts were detected.",
        )
    if int(enriched["txn_count_10min"]) >= RULE_THRESHOLDS["high_velocity"]:
        trigger(
            "HIGH_VELOCITY",
            f"{int(enriched['txn_count_10min'])} transactions occurred within 10 minutes.",
        )
    if int(enriched["location_change"]) == 1:
        trigger("LOCATION_CHANGE", "A recent transaction location change was detected.")
    if float(enriched["merchant_risk_score"]) >= RULE_THRESHOLDS["merchant_risk"]:
        trigger(
            "HIGH_MERCHANT_RISK",
            "The synthetic merchant risk indicator is high.",
        )
    if int(enriched["international_transaction"]) == 1:
        trigger(
            "INTERNATIONAL_TRANSACTION",
            "The transaction is marked as international.",
        )
    if amount_ratio >= RULE_THRESHOLDS["amount_ratio"]:
        trigger(
            "AMOUNT_DEVIATION",
            f"Amount is {amount_ratio:.1f}× the user's historical average.",
        )
    if int(enriched["hour_of_day"]) in UNUSUAL_HOURS:
        trigger(
            "UNUSUAL_HOUR",
            f"Transaction occurred at an unusual hour ({int(enriched['hour_of_day']):02d}:00).",
        )

    maximum_score = float(sum(RULE_WEIGHTS.values()))
    normalized_score = clamp((raw_score / maximum_score) * 100.0)
    return {
        "rule_risk_score": round(normalized_score, 2),
        "triggered_rules": triggered_rules,
        "reasons": reasons,
        "raw_rule_points": int(raw_score),
        "maximum_rule_points": int(maximum_score),
        "rule_notice": "Prototype risk rules for educational purposes.",
    }
