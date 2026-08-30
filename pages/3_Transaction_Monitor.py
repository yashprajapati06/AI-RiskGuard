"""Filter and inspect saved transaction records."""

from __future__ import annotations

import streamlit as st

from config import ALLOWED_PAYMENT_METHODS
from src.database import get_transactions
from src.ui_helpers import configure_page, ensure_app_ready, render_header
from src.utils import parse_json_list

configure_page("Transaction Monitor", "🧾")
ensure_app_ready()
render_header(
    "🧾 Transaction Monitor",
    "Filter and inspect synthetic analysis records saved in the local SQLite database.",
)

filter_columns = st.columns([1, 1, 1, 1.3])
with filter_columns[0]:
    risk_level = st.selectbox("Risk Level", ["ALL", "LOW", "MEDIUM", "HIGH"])
with filter_columns[1]:
    payment_method = st.selectbox("Payment Method", ["ALL", *ALLOWED_PAYMENT_METHODS])
with filter_columns[2]:
    minimum_score = st.slider("Minimum Risk Score", 0, 100, 0)
with filter_columns[3]:
    identifier_search = st.text_input("Transaction ID contains")

transactions = get_transactions(
    risk_level=risk_level,
    payment_method=payment_method,
    minimum_risk_score=minimum_score,
    transaction_id_search=identifier_search,
)
if transactions.empty:
    filters_are_default = (
        risk_level == "ALL"
        and payment_method == "ALL"
        and minimum_score == 0
        and not identifier_search.strip()
    )
    if filters_are_default:
        st.info(
            "No analyses are stored yet. Use **Transaction Risk Checker** to submit "
            "a demonstration transaction; its anonymous record will appear here.",
            icon="💡",
        )
    else:
        st.info(
            "No records match the current filters. Broaden the risk level or payment "
            "method, lower the minimum score, or clear the Transaction ID search.",
            icon="🔎",
        )
    st.stop()

st.caption(
    f"Showing {len(transactions):,} newest-first record(s) for the current filters."
)

display_columns = [
    "transaction_id",
    "created_at",
    "amount",
    "payment_method",
    "fraud_probability",
    "final_risk_score",
    "risk_level",
    "recommended_action",
]
display_frame = transactions[display_columns].rename(
    columns={
        "transaction_id": "Transaction ID",
        "created_at": "Timestamp (UTC)",
        "amount": "Amount",
        "payment_method": "Payment Method",
        "fraud_probability": "ML Likelihood Estimate",
        "final_risk_score": "Final Risk Score",
        "risk_level": "Risk Level",
        "recommended_action": "Recommended Action",
    }
)
display_frame["ML Likelihood Estimate"] = display_frame["ML Likelihood Estimate"].map(
    lambda value: f"{value * 100:.2f}%"
)
st.dataframe(display_frame, hide_index=True, width="stretch")
st.download_button(
    "Download Filtered Report (CSV)",
    transactions.to_csv(index=False).encode("utf-8"),
    file_name="riskguard_filtered_transactions.csv",
    mime="text/csv",
)
st.caption(
    "The export contains fully synthetic or manually entered demonstration fields. "
    "Do not add real payment credentials or personal data to this workflow. The ML "
    "likelihood estimate is not a calibrated real-world fraud probability."
)

st.subheader("Transaction details")
selected_id = st.selectbox(
    "Select Transaction ID", transactions["transaction_id"].tolist()
)
selected = transactions.loc[transactions["transaction_id"] == selected_id].iloc[0]
detail_left, detail_right = st.columns(2)
with detail_left:
    st.json(
        {
            "transaction_id": selected["transaction_id"],
            "user_id": selected["user_id"],
            "merchant_id": selected["merchant_id"],
            "amount": selected["amount"],
            "payment_method": selected["payment_method"],
            "device_type": selected["device_type"],
            "created_at": selected["created_at"],
        }
    )
with detail_right:
    reasons = parse_json_list(selected["risk_reasons"])
    rules = parse_json_list(selected["triggered_rules"])
    st.markdown(f"**Risk level:** {selected['risk_level']}")
    st.markdown(f"**Final combined score:** {selected['final_risk_score']:.2f} / 100")
    st.markdown(f"**Recommended action:** {selected['recommended_action']}")
    st.markdown("**Triggered factors:** " + (", ".join(rules) if rules else "None"))
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.caption("No transparent rule factors were recorded for this transaction.")
