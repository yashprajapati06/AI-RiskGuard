"""Shared Streamlit presentation helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config import DISCLAIMER
from src.bootstrap import initialize_project

RISK_COLORS = {"LOW": "#16a34a", "MEDIUM": "#d97706", "HIGH": "#dc2626"}
RISK_ICONS = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}


def configure_page(title: str, icon: str = "🛡️") -> None:
    """Set consistent page metadata and lightweight styling."""
    st.set_page_config(
        page_title=f"{title} | AI RiskGuard", page_icon=icon, layout="wide"
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px;}
        [data-testid="stMetric"] {
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 0.75rem;
            padding: 0.85rem;
        }
        .risk-card {
            border-radius: 0.75rem;
            padding: 1rem 1.2rem;
            border-left: 6px solid var(--risk-color);
            background: rgba(148, 163, 184, 0.08);
            margin: 0.7rem 0 1rem;
        }
        .small-note {color: #64748b; font-size: 0.88rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def ensure_app_ready() -> dict[str, Any]:
    """Initialize local artifacts once per Streamlit process."""
    return initialize_project()


def render_header(title: str, description: str) -> None:
    """Render a consistent page heading and brand/scope disclaimer."""
    st.title(title)
    st.caption(description)
    st.info(DISCLAIMER, icon="ℹ️")


def render_risk_status(level: str, action: str) -> None:
    """Show a recommendation using icon, text, and color—not color alone."""
    safe_level = level if level in RISK_COLORS else "MEDIUM"
    st.markdown(
        f"""
        <div class="risk-card" style="--risk-color: {RISK_COLORS[safe_level]};">
            <h3 style="margin: 0 0 .35rem;">{RISK_ICONS[safe_level]} {safe_level} RISK</h3>
            <div><strong>Recommended action:</strong> {action}</div>
            <div class="small-note" style="margin-top: .35rem;">
                Prototype recommendation only—no payment is approved, declined, or blocked.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_initialization_notice(status: dict[str, Any]) -> None:
    """Explain only initialization work that occurred during this launch."""
    if status.get("generated_data"):
        st.success("Synthetic dataset generated with the fixed random seed.")
    if status.get("trained_model"):
        st.success(
            "Model artifacts were missing or invalid, so both models were trained and compared."
        )
