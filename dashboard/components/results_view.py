"""
Decision banner, probability gauge, and operational risk metrics component.
"""
import streamlit as st
from typing import Dict, Any


def render_results(result: Dict[str, Any]):
    """Display FastAPI prediction response with visual decision indicators."""
    st.subheader("📊 Scoring Decision & Risk Evaluation")

    decision = result.get("decision", "UNKNOWN")
    prob = float(result.get("fraud_probability", 0.0))
    threshold = float(result.get("threshold", 0.35))
    risk_level = result.get("risk_level", "LOW")
    latency = result.get("latency_ms")
    tx_id = result.get("transaction_id", "N/A")

    # Decision Banner
    if decision == "FRAUD":
        st.error(
            f"### 🚨 FRAUD DETECTED\n"
            f"Transaction **{tx_id}** exceeds operational threshold (`{prob*100:.2f}%` ≥ `τ = {threshold*100:.1f}%`)."
        )
    else:
        st.success(
            f"### ✅ LEGITIMATE TRANSACTION\n"
            f"Transaction **{tx_id}** cleared below operational threshold (`{prob*100:.4f}%` < `τ = {threshold*100:.1f}%`)."
        )

    # Key Metrics Overview
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="Fraud Probability",
            value=f"{prob * 100:.2f}%",
            delta=f"{(prob - threshold) * 100:+.2f}% vs Threshold" if prob >= threshold else None,
            delta_color="inverse"
        )
    with m_col2:
        st.metric(label="Automated Decision", value=decision)
    with m_col3:
        st.metric(label="Risk Tier", value=risk_level)
    with m_col4:
        latency_str = f"{latency:.1f} ms" if latency is not None else "N/A"
        st.metric(label="API Latency", value=latency_str)

    # Visual Probability Bar
    st.markdown("**Probability Gauge vs Operating Threshold**")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Current Probability: **{prob*100:.2f}%** | Operational Decision Threshold: **τ = {threshold*100:.1f}%**")
