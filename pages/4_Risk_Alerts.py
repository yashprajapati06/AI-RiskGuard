"""High-risk alert review page."""

from __future__ import annotations

import streamlit as st

from src.database import get_alerts
from src.ui_helpers import configure_page, ensure_app_ready, render_header

configure_page("Risk Alerts", "🚨")
ensure_app_ready()
render_header(
    "🚨 High-Risk Alerts",
    "Manual-review recommendations created when a saved combined score reaches HIGH risk.",
)

alerts = get_alerts()
if alerts.empty:
    st.success("No HIGH-risk manual-review recommendations are currently stored.")
    st.caption(
        "An alert is created only after a saved analysis reaches 70 or above. Use "
        "the High demonstration preset to show this workflow during evaluation."
    )
    st.stop()

st.metric("Total High-Risk Alerts", f"{len(alerts):,}")
st.dataframe(
    alerts[
        ["transaction_id", "risk_score", "risk_level", "reason", "created_at"]
    ].rename(
        columns={
            "transaction_id": "Transaction ID",
            "risk_score": "Risk Score",
            "risk_level": "Risk Level",
            "reason": "Reason",
            "created_at": "Created At (UTC)",
        }
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Selected alert review")
selected_id = st.selectbox("Select Alert", alerts["transaction_id"].tolist())
alert = alerts.loc[alerts["transaction_id"] == selected_id].iloc[0]
st.error(f"🚨 HIGH RISK · {alert['risk_score']:.2f} / 100")
st.markdown(f"**Transaction:** {alert['transaction_id']}")
st.markdown(f"**Triggered risk explanation:** {alert['reason']}")
st.caption(
    "Recommended next step: immediate manual review. This educational prototype "
    "does not block, authorize, settle, or decline any payment."
)
