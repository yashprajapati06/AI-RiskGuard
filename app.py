"""AI RiskGuard Streamlit entry point."""

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
    "🛡️ AI RiskGuard",
    "A local, explainable digital payment risk management prototype",
)

needs_training = not model_artifacts_are_valid()
needs_data = not DATA_PATH.exists()
if needs_data:
    st.warning("Synthetic dataset not found. Generating it now…")
if needs_training:
    st.warning("Model artifacts not found. Training both candidate models now…")

with st.spinner(
    "Preparing the local project…" if needs_data or needs_training else "Loading…"
):
    initialization_status = ensure_app_ready()
display_initialization_notice(initialization_status)

st.subheader("Project at a glance")
columns = st.columns(4)
columns[0].metric(
    "ML Models Compared", "2", help="Logistic Regression and Random Forest"
)
columns[1].metric(
    "Validated Inputs", "13", help="Behavioral fields only—no payment credentials"
)
columns[2].metric("Risk Outcomes", "3", help="LOW, MEDIUM, and HIGH")
columns[3].metric(
    "Local Persistence", "SQLite", help="Transactions and HIGH-risk alerts"
)

left, right = st.columns([1.25, 1])
with left:
    st.markdown(
        """
        #### How the system works

        1. **Validate** a synthetic transaction and reject malformed or sensitive input.
        2. **Engineer features** such as amount deviation and unusual-hour indicators.
        3. **Score risk** with the selected ML model and transparent prototype rules.
        4. **Recommend action** using a bounded 0–100 combined score.
        5. **Monitor locally** by saving the result and creating a HIGH-risk alert.

        Anonymous IDs are stored for monitoring but deliberately excluded from ML
        training to avoid meaningless high-cardinality patterns.
        """
    )
with right:
    st.markdown("#### Suggested demonstration path")
    st.markdown(
        """
        Use the sidebar in this order:

        - **Transaction Risk Checker:** compare Normal, Medium, and High presets.
        - **Dashboard:** review KPIs and charts created from saved analyses.
        - **Transaction Monitor:** filter and inspect records.
        - **Risk Alerts:** view HIGH-risk review recommendations.
        - **Model Performance:** explain model selection and evaluation trade-offs.
        """
    )

st.warning(
    "Safe-use boundary: never enter a card number, CVV, OTP, PIN, UPI PIN, bank "
    "password, or authentication secret. This prototype must not be used for real "
    "financial decisions or payment authorization.",
    icon="⚠️",
)
