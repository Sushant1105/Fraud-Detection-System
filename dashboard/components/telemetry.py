"""
Session monitoring telemetry and transaction history component.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any


def init_telemetry_state():
    """Ensure history list exists in session state."""
    if "history" not in st.session_state:
        st.session_state["history"] = []


def record_transaction(result: Dict[str, Any]):
    """Append a completed prediction to session history."""
    init_telemetry_state()
    
    top_driver = "N/A"
    top_features = result.get("top_contributing_features", [])
    if top_features:
        top_driver = f"{top_features[0].get('feature', 'N/A')} ({top_features[0].get('shap_value', 0):+.2f})"

    record = {
        "Timestamp": result.get("timestamp", "N/A"),
        "Transaction ID": result.get("transaction_id", "N/A"),
        "Fraud Probability": f"{float(result.get('fraud_probability', 0.0)) * 100:.2f}%",
        "Decision": result.get("decision", "N/A"),
        "Risk Level": result.get("risk_level", "N/A"),
        "Primary Driver": top_driver,
        "Latency": f"{float(result.get('latency_ms', 0.0)):.1f} ms" if result.get("latency_ms") else "N/A",
    }
    st.session_state["history"].insert(0, record)  # Most recent first


def render_telemetry():
    """Render session KPI cards and recent transactions history table."""
    init_telemetry_state()
    history = st.session_state["history"]

    st.subheader("📈 Live Session Telemetry & Monitoring")
    st.write("Real-time summary of transactions evaluated during the current dashboard session.")

    total_count = len(history)
    fraud_count = sum(1 for tx in history if tx["Decision"] == "FRAUD")
    legit_count = total_count - fraud_count
    fraud_pct = (fraud_count / total_count * 100.0) if total_count > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Transactions Checked", value=total_count)
    with kpi2:
        st.metric(label="Frauds Flagged", value=fraud_count, delta=f"{fraud_count} detected" if fraud_count > 0 else None, delta_color="inverse")
    with kpi3:
        st.metric(label="Legitimate Cleared", value=legit_count)
    with kpi4:
        st.metric(label="Session Fraud Incidence", value=f"{fraud_pct:.1f}%")

    if history:
        st.markdown("**Recent Transaction Stream**")
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear Session Telemetry"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No transactions submitted in this session yet. Submit a transaction above to view live telemetry.")
