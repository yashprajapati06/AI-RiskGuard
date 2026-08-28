"""Interactive transaction risk checker."""

from __future__ import annotations

import uuid

import streamlit as st

from config import (
    ALLOWED_DEVICE_TYPES,
    ALLOWED_PAYMENT_METHODS,
    ML_WEIGHT,
    RULE_WEIGHT,
    SAMPLE_TRANSACTIONS_PATH,
)
from src.database import DuplicateTransactionError, save_analysis
from src.predictor import analyze_transaction
from src.ui_helpers import (
    configure_page,
    ensure_app_ready,
    render_header,
    render_risk_status,
)
from src.utils import read_json
from src.validation import TransactionValidationError, validate_transaction

configure_page("Transaction Risk Checker", "🔎")
ensure_app_ready()
render_header(
    "🔎 Transaction Risk Checker",
    "Analyze a safe synthetic payment event with the saved ML model and explainable rules.",
)


def clear_previous_result() -> None:
    """Hide a result that belongs to the previously selected preset."""
    st.session_state.pop("last_riskguard_result", None)


with st.expander("How the final risk score is calculated"):
    st.markdown(
        f"""
        The model returns a fraud probability, which becomes the **ML risk score**.
        Transparent educational rules produce a separate **rule risk score**.

        `Final score = ({ML_WEIGHT:.0%} × ML score) + ({RULE_WEIGHT:.0%} × rule score)`

        - **LOW:** below 35 — normal monitoring recommendation
        - **MEDIUM:** 35 to below 70 — additional verification or review
        - **HIGH:** 70 or above — immediate manual review recommendation

        This score supports a demonstration workflow; it never authorizes or blocks a payment.
        """
    )

try:
    samples = read_json(SAMPLE_TRANSACTIONS_PATH)
    if not {"normal", "medium", "high"}.issubset(samples):
        raise ValueError("Sample transaction file is missing a required preset.")
    samples = {name: validate_transaction(samples[name]) for name in samples}
except (OSError, ValueError, TypeError) as exc:
    st.error(
        "The demonstration presets could not be loaded. Verify "
        f"`data/sample_transactions.json` and restart the app. Technical detail: {exc}"
    )
    st.stop()
preset_name = st.selectbox(
    "Demonstration preset",
    ["normal", "medium", "high"],
    format_func=lambda value: value.title(),
    help="Choose a starting example, then change any field before analysis.",
    on_change=clear_previous_result,
)
preset = samples[preset_name]

with st.form("risk_checker_form", clear_on_submit=False):
    st.markdown("#### Transaction details")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input(
            "Amount (₹)",
            min_value=0.01,
            value=float(preset["amount"]),
            step=100.0,
            key=f"amount_{preset_name}",
        )
        payment_method = st.selectbox(
            "Payment Method",
            ALLOWED_PAYMENT_METHODS,
            index=ALLOWED_PAYMENT_METHODS.index(preset["payment_method"]),
            key=f"payment_method_{preset_name}",
        )
        device_type = st.selectbox(
            "Device Type",
            ALLOWED_DEVICE_TYPES,
            index=ALLOWED_DEVICE_TYPES.index(preset["device_type"]),
            key=f"device_type_{preset_name}",
        )
        is_new_device = st.checkbox(
            "New Device",
            value=bool(preset["is_new_device"]),
            key=f"new_device_{preset_name}",
        )
        previous_failed_txns = st.number_input(
            "Previous Failed Transactions",
            min_value=0,
            value=int(preset["previous_failed_txns"]),
            step=1,
            key=f"failed_{preset_name}",
        )
    with col2:
        txn_count_10min = st.number_input(
            "Transactions in Last 10 Minutes",
            min_value=0,
            value=int(preset["txn_count_10min"]),
            step=1,
            key=f"velocity_{preset_name}",
        )
        average_amount = st.number_input(
            "Average User Transaction Amount (₹)",
            min_value=0.01,
            value=float(preset["avg_user_transaction_amount"]),
            step=100.0,
            key=f"average_amount_{preset_name}",
            help="Synthetic historical average used to calculate amount deviation.",
        )
        location_change = st.checkbox(
            "Location Change",
            value=bool(preset["location_change"]),
            key=f"location_change_{preset_name}",
        )
        merchant_risk_score = st.slider(
            "Merchant Risk Score",
            0.0,
            1.0,
            float(preset["merchant_risk_score"]),
            0.01,
            key=f"merchant_risk_{preset_name}",
            help="Educational synthetic indicator from 0 (lower risk) to 1 (higher risk).",
        )
    with col3:
        account_age_days = st.number_input(
            "Account Age (days)",
            min_value=0,
            value=int(preset["account_age_days"]),
            step=1,
            key=f"account_age_{preset_name}",
        )
        hour_of_day = st.slider(
            "Hour of Day", 0, 23, int(preset["hour_of_day"]), key=f"hour_{preset_name}"
        )
        is_weekend = st.checkbox(
            "Weekend", value=bool(preset["is_weekend"]), key=f"weekend_{preset_name}"
        )
        international_transaction = st.checkbox(
            "International Transaction",
            value=bool(preset["international_transaction"]),
            key=f"international_{preset_name}",
        )
    submitted = st.form_submit_button(
        "Analyze Transaction", type="primary", width="stretch"
    )

if submitted:
    transaction = {
        "transaction_id": f"RG-{uuid.uuid4().hex[:12].upper()}",
        "user_id": f"USR-DEMO-{uuid.uuid4().hex[:6].upper()}",
        "merchant_id": f"MER-DEMO-{uuid.uuid4().hex[:6].upper()}",
        "amount": amount,
        "payment_method": payment_method,
        "device_type": device_type,
        "is_new_device": int(is_new_device),
        "previous_failed_txns": previous_failed_txns,
        "txn_count_10min": txn_count_10min,
        "avg_user_transaction_amount": average_amount,
        "location_change": int(location_change),
        "merchant_risk_score": merchant_risk_score,
        "account_age_days": account_age_days,
        "hour_of_day": hour_of_day,
        "is_weekend": int(is_weekend),
        "international_transaction": int(international_transaction),
    }
    try:
        with st.spinner("Analyzing with the trained model and prototype rules…"):
            analysis = analyze_transaction(transaction)
            save_analysis(transaction, analysis)
        st.session_state["last_riskguard_result"] = {
            "transaction": transaction,
            "analysis": analysis,
        }
        st.success(
            "Analysis completed. The recommendation and anonymous monitoring record "
            "were saved to the local SQLite database."
        )
    except TransactionValidationError as exc:
        for error in exc.errors:
            st.error(error)
    except DuplicateTransactionError as exc:
        st.error(f"This anonymous transaction was already recorded. {exc}")
    except (RuntimeError, OSError) as exc:
        st.error(
            "The analysis service could not complete the request. Run "
            f"`python -m src.train_model` and try again. Technical detail: {exc}"
        )

result = st.session_state.get("last_riskguard_result")
if result:
    analysis = result["analysis"]
    transaction = result["transaction"]
    st.divider()
    render_risk_status(analysis["risk_level"], analysis["recommended_action"])
    metrics = st.columns(4)
    metrics[0].metric(
        "Fraud Probability", f"{analysis['fraud_probability'] * 100:.2f}%"
    )
    metrics[1].metric("ML Risk Score", f"{analysis['ml_risk_score']:.2f} / 100")
    metrics[2].metric("Rule Risk Score", f"{analysis['rule_risk_score']:.2f} / 100")
    metrics[3].metric("Final Risk Score", f"{analysis['final_risk_score']:.2f} / 100")
    st.caption(
        f"Calculation used: {ML_WEIGHT:.0%} × {analysis['ml_risk_score']:.2f} + "
        f"{RULE_WEIGHT:.0%} × {analysis['rule_risk_score']:.2f} = "
        f"{analysis['final_risk_score']:.2f}."
    )

    st.markdown("#### Why was this recommendation produced?")
    if analysis["triggered_rules"]:
        st.write(
            " · ".join(
                code.replace("_", " ").title() for code in analysis["triggered_rules"]
            )
        )
        for reason in analysis["risk_reasons"]:
            st.markdown(f"- {reason}")
    else:
        st.success(
            "No prototype rules were triggered. The ML model may still assign a "
            "non-zero probability because it evaluates the full feature pattern."
        )
    st.caption(
        "Triggered factors explain the rule component. They are not a causal or exact "
        "explanation of every internal ML calculation."
    )
    with st.expander("Anonymous record details"):
        st.json(
            {
                "transaction_id": transaction["transaction_id"],
                "user_id": transaction["user_id"],
                "merchant_id": transaction["merchant_id"],
                "model_prediction": analysis["model_prediction"],
            }
        )
