"""
Streamlit Web Dashboard for Real-Time Fraud Detection & Monitoring.
Acts purely as an interactive client for the FastAPI backend microservice.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dashboard.services.api_client import FraudApiClient
from dashboard.components.header import render_header
from dashboard.components.transaction_form import render_transaction_form
from dashboard.components.results_view import render_results
from dashboard.components.shap_charts import render_shap_explanations
from dashboard.components.telemetry import render_telemetry, record_transaction
from dashboard.components.history_view import render_history_view

# Page Configuration
st.set_page_config(
    page_title="Real-Time Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def main():
    # Initialize API Client
    client = FraudApiClient()

    # Query Backend Health
    is_online, health_data, error_msg = client.check_health()

    # 1. Render Header & System Badge
    render_header(is_online, health_data, error_msg)

    # 2. Tabs: Real-Time Scoring & Database History
    tab_scoring, tab_history = st.tabs([
        "⚡ Real-Time Scoring",
        "🗄️ Database & Transaction History",
    ])

    with tab_scoring:
        # Form and Results
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            submitted, payload = render_transaction_form()

            if submitted and payload:
                if not is_online:
                    st.error("❌ Cannot submit transaction: FastAPI backend service is currently offline.")
                else:
                    with st.spinner("Communicating with FastAPI /predict endpoint..."):
                        success, result, err = client.predict_transaction(payload)
                        if success and result:
                            st.session_state["latest_result"] = result
                            record_transaction(result)
                            st.toast(f"Transaction {result.get('transaction_id', '')} scored successfully!", icon="✅")
                        else:
                            st.error(f"API Error: {err}")

        with col_output:
            if "latest_result" in st.session_state:
                result = st.session_state["latest_result"]
                # Render Decision & Risk Gauge
                render_results(result)
                st.write("")
                # Render SHAP Feature Attribution
                top_features = result.get("top_contributing_features", [])
                render_shap_explanations(top_features)
            else:
                st.info(
                    "👈 **Awaiting Transaction Submission**\n\n"
                    "Select a preset from the intake form on the left or customize transaction values, "
                    "then click **'Check Transaction'** to view automated scoring and SHAP explainability."
                )

        # Session Telemetry & Transaction Log
        st.divider()
        render_telemetry()

    with tab_history:
        render_history_view(client)


if __name__ == "__main__":
    main()

