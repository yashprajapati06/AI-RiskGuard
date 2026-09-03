"""Features shared by training and prediction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from config import (
    AMOUNT_RATIO_EPSILON,
    FEATURE_FAILED_ATTEMPT_THRESHOLD,
    FEATURE_HIGH_AMOUNT_THRESHOLD,
    FEATURE_HIGH_VELOCITY_THRESHOLD,
    UNUSUAL_HOURS,
)


def engineer_features(data: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    """Add the derived risk features to a copy of the input."""
    if isinstance(data, Mapping):
        dataframe = pd.DataFrame([dict(data)])
    elif isinstance(data, pd.DataFrame):
        dataframe = data.copy()
    else:
        raise TypeError("data must be a pandas DataFrame or transaction mapping")

    required = {
        "amount",
        "avg_user_transaction_amount",
        "previous_failed_txns",
        "txn_count_10min",
        "hour_of_day",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(
            "Cannot engineer features; missing columns: " + ", ".join(missing)
        )

    amount = pd.to_numeric(dataframe["amount"], errors="coerce")
    average = pd.to_numeric(
        dataframe["avg_user_transaction_amount"], errors="coerce"
    ).fillna(0.0)
    safe_average = np.maximum(average.to_numpy(dtype=float), AMOUNT_RATIO_EPSILON)
    dataframe["amount_ratio"] = amount.to_numpy(dtype=float) / safe_average
    dataframe["is_high_amount"] = (amount >= FEATURE_HIGH_AMOUNT_THRESHOLD).astype(int)
    dataframe["is_high_velocity"] = (
        pd.to_numeric(dataframe["txn_count_10min"], errors="coerce")
        >= FEATURE_HIGH_VELOCITY_THRESHOLD
    ).astype(int)
    dataframe["failed_attempt_risk"] = (
        pd.to_numeric(dataframe["previous_failed_txns"], errors="coerce")
        >= FEATURE_FAILED_ATTEMPT_THRESHOLD
    ).astype(int)
    dataframe["unusual_hour"] = (
        pd.to_numeric(dataframe["hour_of_day"], errors="coerce").isin(UNUSUAL_HOURS)
    ).astype(int)
    return dataframe
