"""Shared presentation helpers for the Streamlit app."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from src.bootstrap import initialize_project

RISK_COLORS = {"LOW": "#16805b", "MEDIUM": "#b7791f", "HIGH": "#c2413b"}
CHART_COLORS = ["#176b87", "#3d8f83", "#6f82a8", "#b7791f", "#7b6d8d"]
SCOPE_NOTICE = (
    "Fully synthetic demo data · Independent educational project · "
    "Not for payment authorization"
)


def configure_page(title: str, icon: str = "🛡️") -> None:
    """Apply page metadata and the RiskGuard visual system."""
    st.set_page_config(
        page_title=f"{title} | AI RiskGuard", page_icon=icon, layout="wide"
    )
    st.markdown(
        """
        <style>
        :root {
            --rg-ink: #132235;
            --rg-muted: #637083;
            --rg-line: #dce3ea;
            --rg-surface: #ffffff;
            --rg-canvas: #f6f8fa;
            --rg-brand: #176b87;
            --rg-brand-dark: #10556d;
        }

        html, body {
            font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        [data-testid="stIconMaterial"] {
            font-family: "Material Symbols Outlined" !important;
            font-feature-settings: "liga";
            font-weight: normal;
        }
        [data-testid="stExpander"] [data-testid="stIconMaterial"] {
            font-size: 0 !important;
        }
        [data-testid="stExpander"] [data-testid="stIconMaterial"]::after {
            content: "›";
            font-family: "Segoe UI", sans-serif;
            font-size: 1.3rem;
            line-height: 1;
        }
        .stApp { background: var(--rg-canvas); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1.25rem; }
        .block-container {
            max-width: 1280px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #eef2f5;
            border-right: 1px solid var(--rg-line);
        }
        .rg-sidebar-brand {
            color: var(--rg-ink);
            font-size: 1rem;
            font-weight: 760;
            letter-spacing: -0.015em;
            padding: 0.35rem 0.25rem 0;
        }
        .rg-sidebar-caption {
            color: var(--rg-muted);
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            margin: 0.1rem 0.25rem 1.1rem;
            text-transform: uppercase;
        }
        .rg-sidebar-status {
            border-top: 1px solid var(--rg-line);
            color: var(--rg-muted);
            font-size: 0.72rem;
            line-height: 1.45;
            margin: 1rem 0.25rem 0;
            padding-top: 0.8rem;
        }
        [data-testid="stSidebar"] a.rg-nav-link {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            color: var(--rg-ink) !important;
            display: block;
            font-size: 0.875rem;
            margin: 1px 0;
            padding: 0.5rem 0.65rem;
            text-decoration: none;
        }
        [data-testid="stSidebar"] a.rg-nav-link:hover {
            background: #e4ebef;
            border-color: #d7e0e6;
        }
        .rg-nav-active {
            background: #dde8ed;
            border: 1px solid #cbdde4;
            border-radius: 6px;
            color: var(--rg-brand-dark);
            font-size: 0.875rem;
            font-weight: 650;
            margin: 1px 0;
            padding: 0.5rem 0.65rem;
        }

        h1, h2, h3, h4, h5, h6 { color: var(--rg-ink); }
        h1 {
            font-size: clamp(2rem, 4vw, 2.65rem) !important;
            letter-spacing: -0.035em;
            line-height: 1.08 !important;
            margin: 0.25rem 0 0.4rem !important;
        }
        h2 {
            font-size: 1.45rem !important;
            letter-spacing: -0.018em;
            margin-top: 2.1rem !important;
        }
        h3 { font-size: 1.12rem !important; }
        p, li { line-height: 1.65; }

        .rg-eyebrow {
            color: var(--rg-brand-dark);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.095em;
            text-transform: uppercase;
        }
        .rg-eyebrow-dot {
            background: var(--rg-brand);
            border-radius: 50%;
            display: inline-block;
            height: 7px;
            margin-right: 0.48rem;
            width: 7px;
        }
        .rg-page-description {
            color: var(--rg-muted);
            font-size: 1.02rem;
            line-height: 1.55;
            margin: 0 0 1.15rem;
            max-width: 760px;
        }
        .rg-scope-note {
            align-items: flex-start;
            background: #edf4f6;
            border: 1px solid #d5e4e9;
            border-radius: 7px;
            color: #425468;
            display: flex;
            font-size: 0.82rem;
            gap: 0.7rem;
            line-height: 1.5;
            margin-bottom: 1.25rem;
            padding: 0.72rem 0.85rem;
        }
        .rg-scope-label {
            color: var(--rg-brand-dark);
            flex: 0 0 auto;
            font-size: 0.69rem;
            font-weight: 750;
            letter-spacing: 0.07em;
            padding-top: 0.08rem;
        }

        [data-testid="stMetric"] {
            background: var(--rg-surface);
            border: 1px solid var(--rg-line);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(19, 34, 53, 0.035);
            min-height: 108px;
            padding: 1rem 1.1rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--rg-muted);
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.02em;
        }
        [data-testid="stMetricValue"] {
            color: var(--rg-ink);
            font-size: 1.65rem;
            letter-spacing: -0.025em;
        }

        [data-testid="stAlert"] {
            border-radius: 7px;
            border-width: 1px;
            box-shadow: none;
        }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.66);
            border-color: var(--rg-line);
            border-radius: 7px;
        }
        [data-testid="stDataFrame"] {
            background: var(--rg-surface);
            border: 1px solid var(--rg-line);
            border-radius: 7px;
            overflow: hidden;
        }
        [data-testid="stForm"] {
            background: var(--rg-surface);
            border: 1px solid var(--rg-line);
            border-radius: 8px;
            padding: 1.25rem 1.35rem 1.4rem;
        }
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {
            border-radius: 6px;
        }
        [data-testid="stBaseButton-primary"] {
            background: var(--rg-brand);
            border-color: var(--rg-brand);
            border-radius: 6px;
            font-weight: 650;
        }
        [data-testid="stBaseButton-primary"]:hover {
            background: var(--rg-brand-dark);
            border-color: var(--rg-brand-dark);
        }
        [data-testid="stBaseButton-secondary"] { border-radius: 6px; }
        [data-testid="stPageLink"] a {
            background: var(--rg-brand);
            border: 1px solid var(--rg-brand);
            border-radius: 6px;
            color: #ffffff !important;
            font-weight: 650;
            margin-top: 0.35rem;
            padding: 0.58rem 0.9rem;
            text-decoration: none;
        }
        [data-testid="stPageLink"] a:hover {
            background: var(--rg-brand-dark);
            border-color: var(--rg-brand-dark);
        }
        hr { border-color: var(--rg-line) !important; }

        .risk-card {
            --risk-color: #b7791f;
            background: var(--rg-surface);
            border: 1px solid var(--rg-line);
            border-left: 4px solid var(--risk-color);
            border-radius: 7px;
            margin: 0.7rem 0 1.15rem;
            padding: 1.05rem 1.2rem;
        }
        .risk-label {
            align-items: center;
            color: var(--rg-ink);
            display: flex;
            font-size: 0.78rem;
            font-weight: 750;
            gap: 0.48rem;
            letter-spacing: 0.075em;
        }
        .risk-dot {
            background: var(--risk-color);
            border-radius: 50%;
            display: inline-block;
            height: 9px;
            width: 9px;
        }
        .risk-action {
            color: var(--rg-ink);
            font-size: 1.1rem;
            font-weight: 650;
            margin-top: 0.45rem;
        }
        .small-note {
            color: var(--rg-muted);
            font-size: 0.86rem;
            margin-top: 0.32rem;
        }

        @media (max-width: 740px) {
            .block-container { padding: 1.5rem 1rem 3rem; }
            [data-testid="stForm"] { padding: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar(title)


def render_sidebar(current_page: str) -> None:
    """Build a stable navigation menu with a proper home label."""
    links = (
        ("", "Overview", "Home"),
        ("Dashboard", "Dashboard", "Dashboard"),
        (
            "Transaction_Risk_Checker",
            "Transaction risk checker",
            "Transaction Risk Checker",
        ),
        (
            "Transaction_Monitor",
            "Transaction monitor",
            "Transaction Monitor",
        ),
        ("Risk_Alerts", "Risk alerts", "Risk Alerts"),
        ("Model_Performance", "Model performance", "Model Performance"),
    )
    with st.sidebar:
        st.markdown(
            '<div class="rg-sidebar-brand">AI RiskGuard</div>'
            '<div class="rg-sidebar-caption">Risk operations workspace</div>',
            unsafe_allow_html=True,
        )
        for target, label, page_title in links:
            if current_page == page_title:
                st.markdown(
                    f'<div class="rg-nav-active">{escape(label)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<a class="rg-nav-link" href="{escape(target)}" '
                    f'target="_self">{escape(label)}</a>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div class="rg-sidebar-status">LOCAL DEMO<br>'
            "Synthetic data · SQLite storage</div>",
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner=False)
def ensure_app_ready() -> dict[str, Any]:
    """Initialize local artifacts once per Streamlit process."""
    return initialize_project()


def render_header(title: str, description: str) -> None:
    """Render a consistent product header and required scope notice."""
    st.markdown(
        '<div class="rg-eyebrow"><span class="rg-eyebrow-dot"></span>'
        "AI RiskGuard · Risk operations</div>",
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<p class="rg-page-description">{escape(description)}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rg-scope-note"><span class="rg-scope-label">DEMO DATA</span>'
        f"<span>{escape(SCOPE_NOTICE)}</span></div>",
        unsafe_allow_html=True,
    )


def style_chart(figure: Any) -> Any:
    """Give Plotly charts the same restrained visual treatment."""
    figure.update_layout(
        colorway=CHART_COLORS,
        font={
            "family": 'Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            "color": "#3b4a5c",
            "size": 12,
        },
        hoverlabel={
            "bgcolor": "#132235",
            "bordercolor": "#132235",
            "font": {"color": "#ffffff"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
        },
        margin={"l": 24, "r": 16, "t": 58, "b": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"font": {"color": "#132235", "size": 17}, "x": 0.01},
    )
    figure.update_xaxes(
        gridcolor="#e7ebef", linecolor="#cfd7df", zerolinecolor="#dce3ea"
    )
    figure.update_yaxes(
        gridcolor="#e7ebef", linecolor="#cfd7df", zerolinecolor="#dce3ea"
    )
    return figure


def render_risk_status(level: str, action: str) -> None:
    """Show a recommendation with text and shape as well as color."""
    safe_level = level if level in RISK_COLORS else "MEDIUM"
    st.markdown(
        f"""
        <div class="risk-card" style="--risk-color: {RISK_COLORS[safe_level]};">
            <div class="risk-label">
                <span class="risk-dot"></span>{safe_level} RISK
            </div>
            <div class="risk-action">{escape(action)}</div>
            <div class="small-note">
                Review guidance only. This prototype does not approve, decline, or block payments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_initialization_notice(status: dict[str, Any]) -> None:
    """Report setup work performed during this launch."""
    if status.get("generated_data"):
        st.success("The synthetic dataset is ready.")
    if status.get("trained_model"):
        st.success("The model artifacts were rebuilt and validated for this session.")
