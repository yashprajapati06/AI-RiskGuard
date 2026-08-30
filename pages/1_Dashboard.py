"""Live transaction monitoring dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import HIGH_RISK_THRESHOLD
from src.database import get_alerts, get_dashboard_summary, get_transactions
from src.ui_helpers import RISK_COLORS, configure_page, ensure_app_ready, render_header

configure_page("Dashboard", "📊")
ensure_app_ready()
render_header(
    "📊 Monitoring Dashboard",
    "Operational summary of synthetic transaction analyses stored in local SQLite.",
)

summary = get_dashboard_summary()
overview_columns = st.columns(3)
overview_columns[0].metric("Total Analyzed Transactions", f"{summary['total']:,}")
overview_columns[1].metric(
    "Average Combined Risk", f"{summary['average_risk']:.1f} / 100"
)
overview_columns[2].metric("High-Risk Share", f"{summary['high_risk_percentage']:.1f}%")

level_columns = st.columns(3)
level_columns[0].metric("✅ LOW Risk", f"{summary['low_count']:,}")
level_columns[1].metric("⚠️ MEDIUM Risk", f"{summary['medium_count']:,}")
level_columns[2].metric("🚨 HIGH Risk", f"{summary['high_count']:,}")
st.caption(
    "These counts describe model recommendations, not confirmed fraud outcomes. "
    "Ground-truth labels are not available for manually analyzed transactions."
)

transactions = get_transactions()
if transactions.empty:
    st.info(
        "The monitoring database is ready but currently empty. Open **Transaction "
        "Risk Checker**, choose a demonstration preset, and select **Analyze "
        "Transaction**. The saved result will populate these KPIs and charts.",
        icon="💡",
    )
    st.stop()

transactions["created_at"] = pd.to_datetime(transactions["created_at"], utc=True)
transactions["date"] = transactions["created_at"].dt.date

chart_left, chart_right = st.columns(2)
with chart_left:
    risk_counts = (
        transactions["risk_level"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        .rename_axis("risk_level")
        .reset_index(name="count")
    )
    figure = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        title="Risk-level distribution",
        labels={"risk_level": "Risk level", "count": "Transactions"},
    )
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, width="stretch")

with chart_right:
    method_counts = transactions["payment_method"].value_counts().reset_index()
    method_counts.columns = ["payment_method", "count"]
    figure = px.pie(
        method_counts,
        names="payment_method",
        values="count",
        hole=0.45,
        title="Transactions by payment method",
    )
    st.plotly_chart(figure, width="stretch")

chart_left, chart_right = st.columns(2)
with chart_left:
    figure = px.histogram(
        transactions,
        x="final_risk_score",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        nbins=20,
        title="Risk score distribution",
        labels={"final_risk_score": "Final risk score", "count": "Transactions"},
    )
    st.plotly_chart(figure, width="stretch")

with chart_right:
    daily = transactions.groupby("date").size().reset_index(name="transactions")
    figure = px.line(
        daily,
        x="date",
        y="transactions",
        markers=True,
        title="Transaction analysis trend",
        labels={"date": "Date", "transactions": "Transactions"},
    )
    st.plotly_chart(figure, width="stretch")

high_daily = (
    transactions.loc[transactions["risk_level"] == "HIGH"]
    .groupby("date")
    .size()
    .reset_index(name="high_risk_transactions")
)
if not high_daily.empty:
    figure = px.area(
        high_daily,
        x="date",
        y="high_risk_transactions",
        markers=True,
        title="High-risk transactions over time",
        labels={"date": "Date", "high_risk_transactions": "High-risk transactions"},
        color_discrete_sequence=[RISK_COLORS["HIGH"]],
    )
    st.plotly_chart(figure, width="stretch")

st.subheader("Recent high-risk alerts")
alerts = get_alerts(limit=10)
if alerts.empty:
    st.success(
        "No HIGH-risk review alerts are stored. Alerts appear only when a saved "
        f"combined score reaches {HIGH_RISK_THRESHOLD:g} or above."
    )
else:
    alert_display = alerts[
        ["transaction_id", "risk_score", "risk_level", "reason", "created_at"]
    ].rename(
        columns={
            "transaction_id": "Transaction ID",
            "risk_score": "Risk Score",
            "risk_level": "Risk Level",
            "reason": "Triggered Explanation",
            "created_at": "Created At (UTC)",
        }
    )
    st.dataframe(
        alert_display,
        hide_index=True,
        width="stretch",
    )

with st.expander("How to interpret this dashboard"):
    st.markdown(
        """
        - **Risk distribution** shows recommendation volume by LOW, MEDIUM, and HIGH.
        - **Payment method analysis** describes the saved synthetic inputs; it does
          not prove that one payment method causes fraud.
        - **Risk score distribution** helps identify whether results cluster near a
          decision threshold.
        - **Trends** show when analyses were performed, not real payment traffic.
        """
    )
