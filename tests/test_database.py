"""SQLite integration tests using an isolated temporary database."""

import pytest

from src.database import (
    DuplicateTransactionError,
    get_alerts,
    get_dashboard_summary,
    get_transactions,
    save_analysis,
)
from src.predictor import analyze_transaction


def test_empty_database_returns_safe_empty_state(tmp_path) -> None:
    database_path = tmp_path / "empty_riskguard.db"
    assert get_transactions(database_path=database_path).empty
    assert get_alerts(database_path=database_path).empty
    assert get_dashboard_summary(database_path=database_path) == {
        "total": 0,
        "low_count": 0,
        "medium_count": 0,
        "high_count": 0,
        "average_risk": 0.0,
        "high_risk_percentage": 0.0,
    }


def test_high_risk_analysis_creates_one_alert(tmp_path, high_transaction: dict) -> None:
    database_path = tmp_path / "test_riskguard.db"
    transaction = {
        **high_transaction,
        "transaction_id": "TEST-HIGH-001",
        "user_id": "USR-TEST",
        "merchant_id": "MER-TEST",
    }
    analysis = analyze_transaction(high_transaction)
    save_analysis(transaction, analysis, database_path)

    records = get_transactions(database_path=database_path)
    alerts = get_alerts(database_path=database_path)
    summary = get_dashboard_summary(database_path=database_path)
    assert len(records) == 1
    assert len(alerts) == 1
    assert summary["high_count"] == 1

    with pytest.raises(DuplicateTransactionError):
        save_analysis(transaction, analysis, database_path)
    assert len(get_alerts(database_path=database_path)) == 1


def test_inconsistent_analysis_is_rejected(tmp_path, normal_transaction: dict) -> None:
    database_path = tmp_path / "invalid_riskguard.db"
    transaction = {
        **normal_transaction,
        "transaction_id": "TEST-INVALID-001",
        "user_id": "USR-TEST",
        "merchant_id": "MER-TEST",
    }
    analysis = analyze_transaction(normal_transaction)
    analysis["risk_level"] = "HIGH"
    with pytest.raises(ValueError, match="inconsistent"):
        save_analysis(transaction, analysis, database_path)
