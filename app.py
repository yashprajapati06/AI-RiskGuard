"""Main entry point for the RiskGuard workspace."""

from __future__ import annotations

import streamlit as st

from config import DATA_PATH
from src.bootstrap import model_artifacts_are_valid
from src.ui_helpers import (
    configure_page,
    display_initialization_notice,
    ensure_app_ready,
    render_header,
)

configure_page("Home")
render_header(
    "AI RiskGuard",
    "Review transaction risk signals, model output, and manual-review alerts in one local workspace.",
)

needs_training = not model_artifacts_are_valid()
needs_data = not DATA_PATH.exists()
if needs_data:
    st.warning("The training dataset is missing. A local synthetic copy is being prepared.")
if needs_training:
    st.warning("The saved model is unavailable. Both candidate models are being retrained.")

with st.spinner(
    "Preparing the workspace…" if needs_data or needs_training else "Loading workspace…"
):
    initialization_status = ensure_app_ready()
display_initialization_notice(initialization_status)

st.subheader("Workspace overview")
columns = st.columns(4)
columns[0].metric(
    "Models evaluated", "2", help="Logistic Regression and Random Forest"
)
columns[1].metric(
    "Validated inputs", "13", help="Behavioral fields only—no payment credentials"
)
columns[2].metric("Risk bands", "3", help="LOW, MEDIUM, and HIGH")
columns[3].metric(
    "Local data store", "SQLite", help="Transactions and HIGH-risk alerts"
)

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown(
        """
        #### From transaction to review

        Each check follows the same short workflow:

        1. Validate the transaction fields and reject sensitive inputs.
        2. Build behavioral signals such as amount deviation and unusual-hour activity.
        3. Combine the model estimate with a transparent rule score.
        4. Assign a LOW, MEDIUM, or HIGH review recommendation.
        5. Save the anonymous result for local monitoring.

        Anonymous IDs remain available for record lookup, but they are kept out of
        model training because they carry no reliable behavioral meaning.
        """
    )
with right:
    st.markdown("#### Start a review")
    st.markdown(
        """
        Open **Transaction Risk Checker** to assess one of the three sample profiles
        or adjust the fields yourself. Saved assessments appear automatically across
        the rest of the workspace.

        - **Dashboard** gives a quick operational view.
        - **Transaction Monitor** lets you filter and inspect records.
        - **Risk Alerts** collects HIGH-risk recommendations.
        - **Model Performance** documents training and evaluation results.
        """
    )
    st.page_link(
        "pages/2_Transaction_Risk_Checker.py",
        label="Start a transaction check",
    )

st.warning(
    "Use demonstration data only. Never enter a card number, CVV, OTP, PIN, UPI "
    "PIN, bank password, or authentication secret. This prototype must not be used "
    "for real financial decisions or payment authorization."
)
