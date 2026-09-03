"""SQLite storage for transaction results and alerts."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATABASE_PATH, RISK_ACTIONS, ensure_directories
from src.risk_engine import classify_risk
from src.validation import validate_transaction

LOGGER = logging.getLogger(__name__)


class DuplicateTransactionError(ValueError):
    """Raised when a transaction ID is already stored."""


@contextmanager
def database_connection(
    database_path: Path | str = DATABASE_PATH,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection and close it afterward."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def initialize_database(database_path: Path | str = DATABASE_PATH) -> None:
    """Create missing tables and indexes."""
    ensure_directories()
    with database_connection(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                merchant_id TEXT NOT NULL,
                amount REAL NOT NULL,
                payment_method TEXT NOT NULL,
                device_type TEXT NOT NULL,
                is_new_device INTEGER NOT NULL,
                previous_failed_txns INTEGER NOT NULL,
                txn_count_10min INTEGER NOT NULL,
                avg_user_transaction_amount REAL NOT NULL,
                location_change INTEGER NOT NULL,
                merchant_risk_score REAL NOT NULL,
                account_age_days INTEGER NOT NULL,
                hour_of_day INTEGER NOT NULL,
                is_weekend INTEGER NOT NULL,
                international_transaction INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                ml_risk_score REAL NOT NULL,
                rule_risk_score REAL NOT NULL,
                final_risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
                recommended_action TEXT NOT NULL,
                model_prediction INTEGER NOT NULL,
                triggered_rules TEXT NOT NULL DEFAULT '[]',
                risk_reasons TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL UNIQUE,
                risk_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            );

            CREATE INDEX IF NOT EXISTS idx_transactions_created_at
                ON transactions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_transactions_risk_level
                ON transactions(risk_level);
            CREATE INDEX IF NOT EXISTS idx_alerts_created_at
                ON alerts(created_at DESC);
            """
        )
        connection.commit()


def save_analysis(
    transaction: Mapping[str, Any],
    analysis: Mapping[str, Any],
    database_path: Path | str = DATABASE_PATH,
) -> None:
    """Save an analysis and create an alert for a HIGH result.

    Validate again here so callers outside the UI cannot store invalid scores.
    """
    validate_transaction(transaction)
    for identifier in ("transaction_id", "user_id", "merchant_id"):
        if not str(transaction.get(identifier, "")).strip():
            raise ValueError(f"{identifier} cannot be blank when saving an analysis.")
    probability = float(analysis["fraud_probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("fraud_probability must be finite and between 0 and 1.")
    for score_name in ("ml_risk_score", "rule_risk_score", "final_risk_score"):
        score = float(analysis[score_name])
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f"{score_name} must be finite and between 0 and 100.")
    risk_level = str(analysis["risk_level"])
    if risk_level != classify_risk(float(analysis["final_risk_score"])):
        raise ValueError("risk_level is inconsistent with final_risk_score.")
    if analysis["recommended_action"] != RISK_ACTIONS[risk_level]:
        raise ValueError("recommended_action is inconsistent with risk_level.")
    if int(analysis["model_prediction"]) not in (0, 1):
        raise ValueError("model_prediction must be either 0 or 1.")
    if not isinstance(analysis.get("triggered_rules", []), list) or not isinstance(
        analysis.get("risk_reasons", []), list
    ):
        raise TypeError("triggered_rules and risk_reasons must be lists.")
    initialize_database(database_path)
    created_at = datetime.now(timezone.utc).isoformat()
    values = (
        str(transaction["transaction_id"]),
        str(transaction["user_id"]),
        str(transaction["merchant_id"]),
        float(transaction["amount"]),
        str(transaction["payment_method"]),
        str(transaction["device_type"]),
        int(transaction["is_new_device"]),
        int(transaction["previous_failed_txns"]),
        int(transaction["txn_count_10min"]),
        float(transaction["avg_user_transaction_amount"]),
        int(transaction["location_change"]),
        float(transaction["merchant_risk_score"]),
        int(transaction["account_age_days"]),
        int(transaction["hour_of_day"]),
        int(transaction["is_weekend"]),
        int(transaction["international_transaction"]),
        float(analysis["fraud_probability"]),
        float(analysis["ml_risk_score"]),
        float(analysis["rule_risk_score"]),
        float(analysis["final_risk_score"]),
        str(analysis["risk_level"]),
        str(analysis["recommended_action"]),
        int(analysis["model_prediction"]),
        json.dumps(analysis.get("triggered_rules", [])),
        json.dumps(analysis.get("risk_reasons", [])),
        created_at,
    )
    try:
        with database_connection(database_path) as connection:
            connection.execute("BEGIN")
            connection.execute(
                """
                INSERT INTO transactions (
                    transaction_id, user_id, merchant_id, amount, payment_method,
                    device_type, is_new_device, previous_failed_txns, txn_count_10min,
                    avg_user_transaction_amount, location_change, merchant_risk_score,
                    account_age_days, hour_of_day, is_weekend, international_transaction,
                    fraud_probability, ml_risk_score, rule_risk_score, final_risk_score,
                    risk_level, recommended_action, model_prediction, triggered_rules,
                    risk_reasons, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                values,
            )
            if analysis["risk_level"] == "HIGH":
                reasons = analysis.get("risk_reasons", [])
                reason = "; ".join(reasons) if reasons else "High combined risk score."
                connection.execute(
                    """
                    INSERT OR IGNORE INTO alerts
                        (transaction_id, risk_level, risk_score, reason, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        transaction["transaction_id"],
                        analysis["risk_level"],
                        float(analysis["final_risk_score"]),
                        reason,
                        created_at,
                    ),
                )
            connection.commit()
    except sqlite3.IntegrityError as exc:
        if "UNIQUE constraint failed: transactions.transaction_id" in str(exc):
            raise DuplicateTransactionError(
                f"Transaction ID '{transaction['transaction_id']}' has already been analyzed."
            ) from exc
        LOGGER.exception("Database integrity error")
        raise
    except sqlite3.Error:
        LOGGER.exception("Failed to save transaction analysis")
        raise


def get_transactions(
    risk_level: str | None = None,
    payment_method: str | None = None,
    minimum_risk_score: float = 0.0,
    transaction_id_search: str | None = None,
    limit: int | None = None,
    database_path: Path | str = DATABASE_PATH,
) -> pd.DataFrame:
    """Fetch recent transactions using the supplied filters."""
    initialize_database(database_path)
    conditions = ["final_risk_score >= ?"]
    parameters: list[Any] = [float(minimum_risk_score)]
    if risk_level and risk_level != "ALL":
        conditions.append("risk_level = ?")
        parameters.append(risk_level)
    if payment_method and payment_method != "ALL":
        conditions.append("payment_method = ?")
        parameters.append(payment_method)
    if transaction_id_search:
        conditions.append("transaction_id LIKE ?")
        parameters.append(f"%{transaction_id_search.strip()}%")
    query = "SELECT * FROM transactions WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(max(1, int(limit)))
    with database_connection(database_path) as connection:
        return pd.read_sql_query(query, connection, params=parameters)


def get_alerts(
    limit: int | None = None,
    database_path: Path | str = DATABASE_PATH,
) -> pd.DataFrame:
    """Fetch recent high-risk alerts."""
    initialize_database(database_path)
    query = "SELECT * FROM alerts ORDER BY created_at DESC"
    parameters: list[Any] = []
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(max(1, int(limit)))
    with database_connection(database_path) as connection:
        return pd.read_sql_query(query, connection, params=parameters)


def get_dashboard_summary(database_path: Path | str = DATABASE_PATH) -> dict[str, Any]:
    """Calculate dashboard totals from saved transactions."""
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) AS low_count,
                SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) AS medium_count,
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) AS high_count,
                COALESCE(AVG(final_risk_score), 0) AS average_risk
            FROM transactions
            """
        ).fetchone()
    total = int(row["total"] or 0)
    high_count = int(row["high_count"] or 0)
    return {
        "total": total,
        "low_count": int(row["low_count"] or 0),
        "medium_count": int(row["medium_count"] or 0),
        "high_count": high_count,
        "average_risk": float(row["average_risk"] or 0.0),
        "high_risk_percentage": (high_count / total * 100.0) if total else 0.0,
    }
