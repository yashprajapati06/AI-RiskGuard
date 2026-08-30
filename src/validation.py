"""Validation for manual transactions and training datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from config import (
    ALLOWED_DEVICE_TYPES,
    ALLOWED_PAYMENT_METHODS,
    MODEL_RAW_FEATURES,
    RAW_FEATURES,
    REQUIRED_DATASET_COLUMNS,
)


class TransactionValidationError(ValueError):
    """Raised when a transaction fails one or more validation rules."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


FORBIDDEN_SENSITIVE_FIELDS = {
    "card_number",
    "full_card_number",
    "cvv",
    "otp",
    "pin",
    "upi_pin",
    "bank_password",
    "password",
}


def _number(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
    *,
    integer: bool = False,
) -> float | int | None:
    try:
        value = float(data[field])
        if not pd.notna(value):
            raise ValueError
        if not math.isfinite(value):
            errors.append(f"{field} must be finite.")
            return None
        if integer and not value.is_integer():
            errors.append(f"{field} must be a whole number.")
            return None
        return int(value) if integer else value
    except (KeyError, TypeError, ValueError):
        errors.append(f"{field} must be a valid {'whole ' if integer else ''}number.")
        return None


def validate_transaction(transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one inference transaction.

    Returns a clean copy and raises ``TransactionValidationError`` containing
    user-friendly messages when validation fails.
    """
    if not isinstance(transaction, Mapping):
        raise TransactionValidationError(
            ["Transaction input must be a mapping of fields."]
        )

    errors: list[str] = []
    sensitive = FORBIDDEN_SENSITIVE_FIELDS.intersection(
        str(key).strip().lower() for key in transaction
    )
    if sensitive:
        errors.append(
            "Sensitive payment fields are not accepted: "
            + ", ".join(sorted(sensitive))
            + "."
        )

    missing = [field for field in RAW_FEATURES if field not in transaction]
    if missing:
        errors.append("Missing required fields: " + ", ".join(missing) + ".")
        raise TransactionValidationError(errors)

    clean = dict(transaction)
    amount = _number(transaction, "amount", errors)
    average_amount = _number(transaction, "avg_user_transaction_amount", errors)
    merchant_risk = _number(transaction, "merchant_risk_score", errors)
    account_age = _number(transaction, "account_age_days", errors, integer=True)
    hour = _number(transaction, "hour_of_day", errors, integer=True)
    failed = _number(transaction, "previous_failed_txns", errors, integer=True)
    velocity = _number(transaction, "txn_count_10min", errors, integer=True)

    if amount is not None and amount <= 0:
        errors.append("amount must be greater than 0.")
    if average_amount is not None and average_amount <= 0:
        errors.append("avg_user_transaction_amount must be greater than 0.")
    if merchant_risk is not None and not 0 <= merchant_risk <= 1:
        errors.append("merchant_risk_score must be between 0 and 1.")
    if account_age is not None and account_age < 0:
        errors.append("account_age_days cannot be negative.")
    if hour is not None and not 0 <= hour <= 23:
        errors.append("hour_of_day must be between 0 and 23.")
    if failed is not None and failed < 0:
        errors.append("previous_failed_txns cannot be negative.")
    if velocity is not None and velocity < 0:
        errors.append("txn_count_10min cannot be negative.")

    for field in (
        "is_new_device",
        "location_change",
        "is_weekend",
        "international_transaction",
    ):
        value = _number(transaction, field, errors, integer=True)
        if value is not None and value not in (0, 1):
            errors.append(f"{field} must be either 0 or 1.")
        clean[field] = value

    payment_method = str(transaction.get("payment_method", "")).strip()
    device_type = str(transaction.get("device_type", "")).strip()
    if payment_method not in ALLOWED_PAYMENT_METHODS:
        errors.append(
            "payment_method must be one of: " + ", ".join(ALLOWED_PAYMENT_METHODS) + "."
        )
    if device_type not in ALLOWED_DEVICE_TYPES:
        errors.append(
            "device_type must be one of: " + ", ".join(ALLOWED_DEVICE_TYPES) + "."
        )

    if errors:
        raise TransactionValidationError(errors)

    clean.update(
        {
            "amount": amount,
            "avg_user_transaction_amount": average_amount,
            "merchant_risk_score": merchant_risk,
            "account_age_days": account_age,
            "hour_of_day": hour,
            "previous_failed_txns": failed,
            "txn_count_10min": velocity,
            "payment_method": payment_method,
            "device_type": device_type,
        }
    )
    return clean


def validate_training_dataset(
    dataframe: pd.DataFrame, minimum_rows: int = 1_000
) -> None:
    """Validate schema, size, target values, and basic feature constraints."""
    missing = [
        column for column in REQUIRED_DATASET_COLUMNS if column not in dataframe.columns
    ]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    if len(dataframe) < minimum_rows:
        raise ValueError(
            f"Dataset contains {len(dataframe)} rows; at least {minimum_rows} are required."
        )
    if dataframe["fraud"].isna().any():
        raise ValueError("The fraud target cannot contain missing values.")
    target_values = set(dataframe["fraud"].unique().tolist())
    if not {0, 1}.issubset(target_values):
        raise ValueError("The fraud target must contain both classes 0 and 1.")
    if not target_values.issubset({0, 1}):
        raise ValueError("The fraud target may contain only 0 and 1.")
    if dataframe["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique in the training dataset.")
    for identifier in ("transaction_id", "user_id", "merchant_id"):
        if (
            dataframe[identifier].isna().any()
            or dataframe[identifier].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{identifier} cannot contain missing or blank values.")

    numeric_features: dict[str, pd.Series] = {}
    for feature in MODEL_RAW_FEATURES:
        numeric = pd.to_numeric(dataframe[feature], errors="coerce")
        if numeric.isna().any() or not numeric.map(math.isfinite).all():
            raise ValueError(f"{feature} must contain only finite numeric values.")
        numeric_features[feature] = numeric

    for feature in ("amount", "avg_user_transaction_amount"):
        if (numeric_features[feature] <= 0).any():
            raise ValueError(f"{feature} must be greater than 0.")
    for feature in (
        "previous_failed_txns",
        "txn_count_10min",
        "account_age_days",
    ):
        values = numeric_features[feature]
        if (values < 0).any() or not (values % 1 == 0).all():
            raise ValueError(f"{feature} must contain non-negative whole numbers.")
    hours = numeric_features["hour_of_day"]
    if not hours.between(0, 23).all() or not (hours % 1 == 0).all():
        raise ValueError("hour_of_day must be a whole number between 0 and 23.")
    for feature in ("location_change", "is_weekend", "international_transaction"):
        if not numeric_features[feature].isin((0, 1)).all():
            raise ValueError(f"{feature} must contain only 0 or 1.")
