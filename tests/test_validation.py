"""Tests for safe manual input validation."""

import math

import pytest

from src.validation import TransactionValidationError, validate_transaction


def test_negative_amount_is_rejected(normal_transaction: dict) -> None:
    normal_transaction["amount"] = -1
    with pytest.raises(
        TransactionValidationError, match="amount must be greater than 0"
    ):
        validate_transaction(normal_transaction)


def test_unknown_payment_method_is_rejected(normal_transaction: dict) -> None:
    normal_transaction["payment_method"] = "CrypticPay"
    with pytest.raises(TransactionValidationError, match="payment_method"):
        validate_transaction(normal_transaction)


def test_sensitive_fields_are_rejected(normal_transaction: dict) -> None:
    normal_transaction["cvv"] = "123"
    with pytest.raises(TransactionValidationError, match="Sensitive payment fields"):
        validate_transaction(normal_transaction)


def test_valid_transaction_is_normalized(normal_transaction: dict) -> None:
    normal_transaction["hour_of_day"] = 12.0
    clean = validate_transaction(normal_transaction)
    assert clean["hour_of_day"] == 12
    assert clean["payment_method"] == "UPI"


@pytest.mark.parametrize("invalid_amount", [math.inf, -math.inf, math.nan])
def test_non_finite_amount_is_rejected(
    normal_transaction: dict, invalid_amount: float
) -> None:
    normal_transaction["amount"] = invalid_amount
    with pytest.raises(TransactionValidationError):
        validate_transaction(normal_transaction)
