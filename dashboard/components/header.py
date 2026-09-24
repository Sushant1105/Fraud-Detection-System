"""
Dashboard header and system health status badge component.
"""
import streamlit as st
from typing import Optional, Dict, Any


def render_header(is_online: bool, health_data: Optional[Dict[str, Any]], error_msg: Optional[str]):
    """Render top title bar and live API connectivity indicator."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🛡️ Real-Time Fraud Detection Monitor")
        st.caption("Synchronous transaction risk scoring, automated decisioning, and local SHAP explainability.")

    with col2:
        if is_online and health_data:
            threshold = health_data.get("decision_threshold", "N/A")
            algo = health_data.get("model_algorithm", "ML Model")
            st.success(f"🟢 **API Online**\n\n`{algo}` | `τ = {threshold}`")
        else:
            st.error("🔴 **API Offline**\n\nFastAPI is unreachable")

    if not is_online:
        st.warning(
            f"⚠️ **Backend Unavailable**: {error_msg or 'Connection refused'}.\n\n"
            "Please ensure the FastAPI service is running locally:\n"
            "```bash\n"
            "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload\n"
            "```"
        )
    st.divider()
