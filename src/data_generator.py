"""Synthetic transaction data for local demos and tests."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ALLOWED_DEVICE_TYPES,
    ALLOWED_PAYMENT_METHODS,
    BASE_DIR,
    DATA_PATH,
    DATASET_METADATA_PATH,
    DATASET_SIZE,
    RANDOM_STATE,
    ensure_directories,
)
from src.utils import configure_logging, write_json

LOGGER = logging.getLogger(__name__)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_synthetic_transactions(
    output_path: Path | str = DATA_PATH,
    n_transactions: int = DATASET_SIZE,
    random_state: int = RANDOM_STATE,
    force: bool = False,
) -> pd.DataFrame:
    """Generate and save a repeatable synthetic dataset.

    Labels mix several risk signals with random noise. No real payment or personal
    data is created.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        LOGGER.info("Using existing synthetic dataset at %s", output_path)
        return pd.read_csv(output_path)
    if n_transactions < 1_000:
        raise ValueError(
            "n_transactions must be at least 1,000 for useful training data."
        )

    ensure_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    user_number = rng.integers(1, 3_501, size=n_transactions)
    merchant_number = rng.integers(1, 501, size=n_transactions)

    user_baselines = rng.lognormal(mean=np.log(2_200), sigma=0.65, size=3_500)
    avg_amount = np.clip(user_baselines[user_number - 1], 100, 50_000)
    amount_multiplier = rng.lognormal(mean=-0.05, sigma=0.72, size=n_transactions)
    amount_spike = rng.random(n_transactions) < 0.035
    amount_multiplier[amount_spike] *= rng.uniform(3.0, 10.0, amount_spike.sum())
    amount = np.clip(avg_amount * amount_multiplier, 20, 150_000).round(2)

    payment_method = rng.choice(
        ALLOWED_PAYMENT_METHODS,
        size=n_transactions,
        p=[0.46, 0.29, 0.15, 0.10],
    )
    device_type = rng.choice(
        ALLOWED_DEVICE_TYPES,
        size=n_transactions,
        p=[0.57, 0.24, 0.19],
    )
    is_new_device = rng.binomial(1, 0.12, n_transactions)

    previous_failed_txns = rng.poisson(0.45, n_transactions)
    repeated_failure = rng.random(n_transactions) < 0.025
    previous_failed_txns[repeated_failure] += rng.integers(
        3, 8, size=repeated_failure.sum()
    )
    previous_failed_txns = np.clip(previous_failed_txns, 0, 12)

    txn_count_10min = 1 + rng.poisson(0.75, n_transactions)
    velocity_burst = rng.random(n_transactions) < 0.035
    txn_count_10min[velocity_burst] += rng.integers(4, 11, velocity_burst.sum())
    txn_count_10min = np.clip(txn_count_10min, 1, 20)

    location_change = rng.binomial(1, 0.08, n_transactions)
    merchant_base_risk = rng.beta(2.0, 6.0, size=500)
    risky_merchants = rng.choice(500, size=28, replace=False)
    merchant_base_risk[risky_merchants] = rng.uniform(0.72, 0.98, size=28)
    merchant_risk_score = np.clip(
        merchant_base_risk[merchant_number - 1] + rng.normal(0, 0.025, n_transactions),
        0,
        1,
    ).round(4)

    account_age_days = np.clip(
        rng.gamma(shape=2.2, scale=330, size=n_transactions), 1, 3_650
    ).astype(int)
    hour_weights = np.array(
        [2, 1, 1, 1, 1, 2, 4, 7, 8, 7, 6, 6, 7, 7, 6, 6, 7, 8, 9, 9, 8, 7, 5, 3],
        dtype=float,
    )
    hour_of_day = rng.choice(
        24, size=n_transactions, p=hour_weights / hour_weights.sum()
    )
    is_weekend = rng.binomial(1, 2 / 7, n_transactions)
    international_transaction = rng.binomial(1, 0.055, n_transactions)

    amount_ratio = amount / np.maximum(avg_amount, 1e-6)
    unusual_hour = np.isin(hour_of_day, [0, 1, 2, 3, 4]).astype(int)

    # Noise keeps the labels from becoming a direct copy of the rules.
    log_odds = (
        -5.40
        + 0.90 * np.log1p(amount_ratio)
        + 1.00 * is_new_device
        + 0.40 * np.minimum(previous_failed_txns, 6)
        + 0.28 * np.minimum(txn_count_10min - 1, 8)
        + 1.00 * location_change
        + 1.70 * merchant_risk_score
        + 1.00 * international_transaction
        + 0.90 * unusual_hour
        + 0.55 * (amount >= 25_000)
        + 0.45 * ((is_new_device == 1) & (location_change == 1))
        + 0.40 * ((previous_failed_txns >= 3) & (txn_count_10min >= 5))
        + rng.normal(0, 0.45, n_transactions)
    )
    fraud_probability = _sigmoid(log_odds)
    fraud = rng.binomial(1, fraud_probability, n_transactions)

    dataframe = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:07d}" for i in range(1, n_transactions + 1)],
            "user_id": [f"USR{value:05d}" for value in user_number],
            "merchant_id": [f"MER{value:04d}" for value in merchant_number],
            "amount": amount,
            "payment_method": payment_method,
            "device_type": device_type,
            "is_new_device": is_new_device,
            "previous_failed_txns": previous_failed_txns,
            "txn_count_10min": txn_count_10min,
            "avg_user_transaction_amount": avg_amount.round(2),
            "location_change": location_change,
            "merchant_risk_score": merchant_risk_score,
            "account_age_days": account_age_days,
            "hour_of_day": hour_of_day,
            "is_weekend": is_weekend,
            "international_transaction": international_transaction,
            "fraud": fraud,
        }
    )
    dataframe.to_csv(output_path, index=False)
    if output_path.resolve() == DATA_PATH.resolve():
        write_json(
            DATASET_METADATA_PATH,
            {
                "source_id": "local_synthetic_generator",
                "source_name": "Locally generated synthetic payment transactions",
                "source_url": "local:src.data_generator",
                "data_origin": "locally_generated_synthetic_fallback",
                "source_rows": len(dataframe),
                "sample_rows": len(dataframe),
                "sampling_strategy": "complete_generated_dataset",
                "amount_normalization": "No external currency conversion.",
            },
        )
    distribution = dataframe["fraud"].value_counts().sort_index().to_dict()
    fraud_rate = dataframe["fraud"].mean() * 100
    LOGGER.info(
        "Generated %d synthetic transactions at %s; fraud rate %.2f%%",
        len(dataframe),
        output_path,
        fraud_rate,
    )
    try:
        display_path = output_path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        display_path = output_path.name
    print(f"Generated {len(dataframe):,} transactions at {display_path}")
    print(f"Fraud distribution: {distribution} (fraud rate: {fraud_rate:.2f}%)")
    return dataframe


def load_or_generate_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset, creating it when it is missing."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return generate_synthetic_transactions(path)


if __name__ == "__main__":
    configure_logging()
    generate_synthetic_transactions()
