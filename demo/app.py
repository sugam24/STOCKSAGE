"""
demo/app.py
------------
Days 55–58: Streamlit Interactive Financial Research Dashboard.

Features:
  - Day 55: Streamlit UI Skeleton (Ticker search input & execution controls).
  - Day 56: Connected to FastAPI backend with asynchronous polling.
  - Day 57: Interactive Plotly financial charts (Price trend & RSI gauge with zones).
  - Day 58: Human-in-the-Loop review gate (Inspect intermediate data, Approve/Revise).
  - Day 59: API Key authentication header support.

Usage:
    uv run streamlit run demo/app.py
"""

from __future__ import annotations

import os
import time
from typing import Any
import httpx
import plotly.graph_objects as go
import streamlit as st

def _get_setting(key: str, default: str) -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

API_BASE_URL = _get_setting("STOCKSAGE_API_URL", "http://127.0.0.1:8000")
DEFAULT_API_KEY = _get_setting("STOCKSAGE_API_KEY", "stocksage-dev-key-12345")

# Page Configuration
st.set_page_config(
    page_title="StockSage — Autonomous AI Financial Analyst",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("StockSage Settings")
    api_url = st.text_input("Backend API URL", value=API_BASE_URL)
    api_key = st.text_input("X-API-Key", value=DEFAULT_API_KEY, type="password")
    enable_hitl = st.checkbox("Enable Human-in-the-Loop Review Gate", value=False)
    st.markdown("---")
    st.markdown(
        "**StockSage Architecture**:\n"
        "- 🧠 Supervisor Agent\n"
        "- 📊 Data Agent (Technicals)\n"
        "- 📰 News Agent (10-Q RAG)\n"
        "- ⚖️ Risk Agent (VaR / MDD)\n"
        "- ✍️ Analyst Agent (Synthesis)\n"
        "- 🔍 Critic Node (Reflexion)"
    )


# ---------------------------------------------------------------------------
# Plotly Chart Helpers (Day 57)
# ---------------------------------------------------------------------------

def create_price_chart(ticker: str, prices: list[float]) -> go.Figure:
    """Generate a clean interactive line chart for historical prices."""
    fig = go.Figure()
    days = list(range(len(prices)))
    fig.add_trace(go.Scatter(
        x=days,
        y=prices,
        mode="lines",
        name="Closing Price",
        line=dict(color="#2962FF", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(41, 98, 255, 0.08)",
    ))
    fig.update_layout(
        title=f"{ticker} — 6-Month Price Trajectory",
        xaxis_title="Trading Days",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_rsi_gauge(rsi_val: float) -> go.Figure:
    """Generate an RSI dial gauge highlighting Oversold, Neutral, and Overbought zones."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        title={'text': "RSI (14-Period)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1E293B"},
            'steps': [
                {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.3)"},     # Oversold (Green)
                {'range': [30, 70], 'color': "rgba(226, 232, 240, 0.5)"},  # Neutral (Gray)
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"},   # Overbought (Red)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.8,
                'value': rsi_val,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Main App Header
# ---------------------------------------------------------------------------
st.title("📈 StockSage AI — Equity Research Terminal")
st.caption("Autonomous multi-agent quantitative financial analysis powered by LangGraph & Groq.")

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter Stock Ticker Symbol:", value="AAPL", max_chars=10).upper().strip()
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("🚀 Analyze Ticker", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Session State & Execution Logic (Days 56 & 58)
# ---------------------------------------------------------------------------

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "session_data" not in st.session_state:
    st.session_state.session_data = None


def trigger_backend_analysis(ticker: str, hitl: bool) -> str | None:
    """Launches analysis on the FastAPI backend."""
    headers = {"X-API-Key": api_key}
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{api_url}/v1/analysis",
                json={"ticker": ticker, "enable_hitl": hitl},
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("session_id")
            elif resp.status_code == 401:
                st.error("Authentication failed: Invalid X-API-Key header.")
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
    except Exception as exc:
        st.error(f"Could not connect to FastAPI backend at {api_url}: {exc}")
    return None


def fetch_session_details(session_id: str) -> dict[str, Any] | None:
    """Polls session status from FastAPI backend."""
    headers = {"X-API-Key": api_key}
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{api_url}/v1/analysis/{session_id}", headers=headers)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def submit_human_approval(session_id: str, action: str, feedback: str = ""):
    """Submits human-in-the-loop decision to backend (Day 58)."""
    headers = {"X-API-Key": api_key}
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{api_url}/v1/analysis/{session_id}/approval",
                json={"action": action, "feedback": feedback},
                headers=headers,
            )
    except Exception as exc:
        st.error(f"Error submitting approval: {exc}")


# Handle Trigger
if analyze_btn and ticker_input:
    with st.spinner(f"Contacting StockSage backend for {ticker_input}..."):
        sid = trigger_backend_analysis(ticker_input, enable_hitl)
        if sid:
            st.session_state.active_session_id = sid
            st.session_state.session_data = None


# Polling Loop
if st.session_state.active_session_id:
    sid = st.session_state.active_session_id
    poll_placeholder = st.empty()

    data = fetch_session_details(sid)
    st.session_state.session_data = data

    if data:
        status = data.get("status")

        # -------------------------------------------------------------------
        # In-Progress Spinner
        # -------------------------------------------------------------------
        if status == "processing":
            with poll_placeholder.container():
                st.info(f"⏳ Multi-agent graph working on **{data.get('ticker')}** (Session `{sid[:8]}`)...")
                time.sleep(2)
                st.rerun()

        # -------------------------------------------------------------------
        # Day 58: Human-in-the-Loop Review Gate
        # -------------------------------------------------------------------
        elif status == "pending_approval":
            with poll_placeholder.container():
                st.warning("⚠️ **Human-in-the-Loop Review Required** before generating the final report.")
                st.write("Inspect the preliminary agent findings gathered below:")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.json(data.get("stock_data") or {"info": "Market data pending"})
                with col_b:
                    st.json(data.get("risk_metrics") or {"info": "Risk metrics pending"})

                st.subheader("Your Editorial Decision:")
                app_col1, app_col2 = st.columns([1, 3])
                with app_col1:
                    if st.button("✅ Approve Synthesis", type="primary"):
                        submit_human_approval(sid, "approve")
                        st.rerun()
                with app_col2:
                    rev_feedback = st.text_input("Revision notes (optional):", key="rev_input")
                    if st.button("🔄 Request Revision"):
                        submit_human_approval(sid, f"revise: {rev_feedback}")
                        st.rerun()

        # -------------------------------------------------------------------
        # Completed Dashboard & Visualizations (Days 56 & 57)
        # -------------------------------------------------------------------
        elif status == "completed":
            poll_placeholder.empty()
            st.success(f"✅ Analysis for **{data.get('ticker')}** completed!")

            stock_data = data.get("stock_data") or {}
            risk_metrics = data.get("risk_metrics") or {}
            prices = stock_data.get("price_history") or []
            rsi_val = stock_data.get("rsi")

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            latest_close = stock_data.get("latest_close")
            m1.metric("Latest Close", f"${latest_close:.2f}" if latest_close else "N/A")
            m2.metric("Annualized Volatility", f"{risk_metrics.get('volatility_annualized', 0):.1%}")
            m3.metric("95% 1-Day VaR", f"{risk_metrics.get('var_95_daily', 0):.1%}")
            m4.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.1%}")

            # Day 57 Charts Row
            if prices or (rsi_val is not None):
                st.markdown("---")
                c1, c2 = st.columns([2, 1])
                with c1:
                    if prices and len(prices) > 2:
                        st.plotly_chart(create_price_chart(data.get("ticker"), prices), use_container_width=True)
                    else:
                        st.info("Price history not available for charting.")
                with c2:
                    if rsi_val is not None:
                        st.plotly_chart(create_rsi_gauge(rsi_val), use_container_width=True)
                    else:
                        st.info("RSI indicator not available.")

            # Final Report Presentation
            st.markdown("---")
            st.markdown(data.get("final_report") or "No report content generated.")

            # Raw Evidence Accordion
            with st.expander("📚 View Unstructured Evidence & SEC Filings Context"):
                contexts = data.get("news_context") or []
                if contexts:
                    for item in contexts:
                        st.markdown(f"- {item}")
                else:
                    st.write("No news chunks stored.")

        elif status == "failed":
            poll_placeholder.empty()
            st.error(f"❌ Analysis failed: {data.get('error_log')}")
