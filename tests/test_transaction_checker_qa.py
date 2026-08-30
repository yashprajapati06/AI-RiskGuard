"""End-to-end QA coverage for the Streamlit transaction risk checker."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from config import (
    DATA_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    RISK_ACTIONS,
    RULE_WEIGHTS,
)
from src import database, predictor, ui_helpers
from src.database import (
    DuplicateTransactionError,
    get_alerts,
    get_transactions,
    save_analysis,
)
from src.evaluation import get_fraud_class_index, predict_fraud_probabilities
from src.feature_engineering import engineer_features
from src.predictor import analyze_transaction, load_model_artifacts
from src.risk_engine import calculate_final_risk, classify_risk
from src.utils import read_json
from src.validation import validate_transaction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PAGE = PROJECT_ROOT / "pages" / "2_Transaction_Risk_Checker.py"
QA_SAMPLE_SIZE_PER_CLASS = 10
QA_RANDOM_STATE = 42

REQUIRED_ANALYSIS_KEYS = {
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


def _held_out_random_cases() -> list[tuple[int, dict[str, Any]]]:
    """Build a balanced, seeded QA batch from the chronological test partition.

    The fraud target is used only to select coverage cases. It is deliberately
    omitted from every transaction sent to the inference service. Source-absent
    device and merchant inputs use neutral demonstration values.
    """
    metadata = read_json(MODEL_METADATA_PATH)
    assert metadata["split_strategy"] == "chronological_80_20"
    test_rows = int(metadata["test_rows"])
    data = pd.read_csv(DATA_PATH)
    held_out = data.tail(test_rows)
    selected = pd.concat(
        [
            held_out.loc[held_out["fraud"] == label].sample(
                QA_SAMPLE_SIZE_PER_CLASS,
                random_state=QA_RANDOM_STATE,
            )
            for label in (0, 1)
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=QA_RANDOM_STATE, ignore_index=True)

    assert selected["transaction_id"].is_unique
    cases: list[tuple[int, dict[str, Any]]] = []
    for case_number, row in enumerate(selected.itertuples(index=False), start=1):
        transaction = {
            "transaction_id": f"QA-RANDOM-{case_number:02d}",
            "user_id": f"QA-USER-{case_number:02d}",
            "merchant_id": f"QA-MERCHANT-{case_number:02d}",
            "amount": float(row.amount),
            "payment_method": "Card",
            "device_type": "Web",
            "is_new_device": 0,
            "previous_failed_txns": int(row.previous_failed_txns),
            "txn_count_10min": int(row.txn_count_10min),
            "avg_user_transaction_amount": float(row.avg_user_transaction_amount),
            "location_change": int(row.location_change),
            "merchant_risk_score": 0.0,
            "account_age_days": int(row.account_age_days),
            "hour_of_day": int(row.hour_of_day),
            "is_weekend": int(row.is_weekend),
            "international_transaction": int(row.international_transaction),
        }
        assert "fraud" not in transaction
        assert "event_timestamp" not in transaction
        cases.append((int(row.fraud), validate_transaction(transaction)))
    return cases


def test_twenty_seeded_held_out_transactions_end_to_end(tmp_path: Path) -> None:
    """Check 20 held-out cases through inference, rules, and isolated storage."""
    cases = _held_out_random_cases()
    transactions = [transaction for _, transaction in cases]
    assert len(cases) == 20
    assert {label for label, _ in cases} == {0, 1}
    assert len({item["transaction_id"] for item in transactions}) == 20

    model, preprocessor = load_model_artifacts()
    assert get_fraud_class_index(model) >= 0
    engineered = engineer_features(pd.DataFrame(transactions))
    transformed = preprocessor.transform(engineered[MODEL_FEATURES])
    batch_probabilities = predict_fraud_probabilities(model, transformed)
    batch_predictions = np.asarray(model.predict(transformed), dtype=int)
    assert batch_probabilities.shape == (20,)
    assert np.isfinite(batch_probabilities).all()
    assert ((batch_probabilities >= 0) & (batch_probabilities <= 1)).all()

    database_path = tmp_path / "random-checker-qa.db"
    analyses: list[dict[str, Any]] = []
    for index, transaction in enumerate(transactions):
        analysis = analyze_transaction(transaction)
        analyses.append(analysis)

        assert REQUIRED_ANALYSIS_KEYS.issubset(analysis)
        assert math.isfinite(analysis["fraud_probability"])
        assert 0 <= analysis["fraud_probability"] <= 1
        for score_name in (
            "ml_risk_score",
            "rule_risk_score",
            "final_risk_score",
        ):
            assert math.isfinite(analysis[score_name])
            assert 0 <= analysis[score_name] <= 100
        assert analysis["fraud_probability"] == pytest.approx(
            batch_probabilities[index], abs=5.1e-7
        )
        assert analysis["model_prediction"] == int(batch_predictions[index])
        assert analysis["risk_level"] == classify_risk(analysis["final_risk_score"])
        assert analysis["recommended_action"] == RISK_ACTIONS[analysis["risk_level"]]

        recomputed = calculate_final_risk(
            batch_probabilities[index] * 100.0,
            analysis["rule_risk_score"],
        )
        assert analysis["final_risk_score"] == pytest.approx(
            recomputed["final_risk_score"], abs=0.01
        )
        assert len(analysis["triggered_rules"]) == len(analysis["risk_reasons"])
        assert len(analysis["triggered_rules"]) == len(set(analysis["triggered_rules"]))
        assert set(analysis["triggered_rules"]).issubset(RULE_WEIGHTS)
        save_analysis(transaction, analysis, database_path)

    levels = {analysis["risk_level"] for analysis in analyses}
    assert levels == {"LOW", "MEDIUM", "HIGH"}
    assert get_transactions(database_path=database_path).shape[0] == 20
    expected_alerts = sum(analysis["risk_level"] == "HIGH" for analysis in analyses)
    assert get_alerts(database_path=database_path).shape[0] == expected_alerts

    with pytest.raises(DuplicateTransactionError):
        save_analysis(transactions[0], analyses[0], database_path)
    assert get_transactions(database_path=database_path).shape[0] == 20
    assert get_alerts(database_path=database_path).shape[0] == expected_alerts


def test_streamlit_checker_form_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify widget values reach analysis/storage and the result is rendered."""
    captured: dict[str, Any] = {}
    fake_analysis = {
        "fraud_probability": 0.10,
        "ml_risk_score": 10.0,
        "rule_risk_score": 20.0,
        "final_risk_score": 13.0,
        "risk_level": "LOW",
        "recommended_action": RISK_ACTIONS["LOW"],
        "model_prediction": 0,
        "triggered_rules": ["NEW_DEVICE"],
        "risk_reasons": ["Transaction originated from a new device."],
    }

    def fake_analyze(transaction: dict[str, Any]) -> dict[str, Any]:
        captured["analyzed"] = dict(transaction)
        return dict(fake_analysis)

    def fake_save(transaction: dict[str, Any], analysis: dict[str, Any]) -> None:
        captured["saved_transaction"] = dict(transaction)
        captured["saved_analysis"] = dict(analysis)

    monkeypatch.setattr(ui_helpers, "ensure_app_ready", dict)
    monkeypatch.setattr(predictor, "analyze_transaction", fake_analyze)
    monkeypatch.setattr(database, "save_analysis", fake_save)

    app = AppTest.from_file(str(CHECKER_PAGE), default_timeout=30).run()
    assert not app.exception
    assert not app.error

    app.number_input(key="amount_normal").set_value(1_234.0)
    app.selectbox(key="payment_method_normal").select("Wallet")
    app.selectbox(key="device_type_normal").select("Web")
    app.checkbox(key="new_device_normal").check()
    app.number_input(key="failed_normal").set_value(2)
    app.number_input(key="velocity_normal").set_value(4)
    app.number_input(key="average_amount_normal").set_value(800.0)
    app.checkbox(key="location_change_normal").check()
    app.slider(key="merchant_risk_normal").set_value(0.42)
    app.number_input(key="account_age_normal").set_value(365)
    app.slider(key="hour_normal").set_value(18)
    app.checkbox(key="weekend_normal").check()
    app.checkbox(key="international_normal").check()
    app.button(key="FormSubmitter:risk_checker_form-Analyze Transaction").click().run(
        timeout=30
    )

    assert not app.exception
    assert not app.error
    submitted = captured["analyzed"]
    assert submitted["amount"] == 1_234.0
    assert submitted["payment_method"] == "Wallet"
    assert submitted["device_type"] == "Web"
    assert submitted["is_new_device"] == 1
    assert submitted["previous_failed_txns"] == 2
    assert submitted["txn_count_10min"] == 4
    assert submitted["avg_user_transaction_amount"] == 800.0
    assert submitted["location_change"] == 1
    assert submitted["merchant_risk_score"] == pytest.approx(0.42)
    assert submitted["account_age_days"] == 365
    assert submitted["hour_of_day"] == 18
    assert submitted["is_weekend"] == 1
    assert submitted["international_transaction"] == 1
    assert re.fullmatch(r"RG-[A-F0-9]{12}", submitted["transaction_id"])
    assert re.fullmatch(r"USR-DEMO-[A-F0-9]{6}", submitted["user_id"])
    assert re.fullmatch(r"MER-DEMO-[A-F0-9]{6}", submitted["merchant_id"])
    assert captured["saved_transaction"] == submitted
    assert captured["saved_analysis"] == fake_analysis
    assert any("Analysis completed" in item.value for item in app.success)
    assert any("After changing any field" in item.value for item in app.info)
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("ML Fraud-Likelihood Estimate", "10.00%"),
        ("ML Risk Score", "10.00 / 100"),
        ("Rule Risk Score", "20.00 / 100"),
        ("Final Risk Score", "13.00 / 100"),
    ]
    assert app.session_state["last_riskguard_result"]["analysis"] == fake_analysis
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC",
        app.session_state["last_riskguard_result"]["analyzed_at"],
    )
    assert any("Result refreshed:" in item.value for item in app.caption)
