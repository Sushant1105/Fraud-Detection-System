"""
Transaction input form component with preset quick-load buttons.
"""
import streamlit as st
from typing import Dict, Any, Tuple, Optional
from dashboard.utils.presets import LEGITIMATE_SAMPLE, FRAUD_SAMPLE


def init_form_state():
    """Ensure session state contains form keys."""
    if "form_values" not in st.session_state:
        st.session_state["form_values"] = dict(LEGITIMATE_SAMPLE)


def render_transaction_form() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Render transaction input form and preset selectors.
    Returns: (submitted, transaction_payload_dict)
    """
    init_form_state()

    st.subheader("📝 Transaction Intake Form")
    st.write("Submit raw financial transaction features for real-time scoring and SHAP analysis.")

    # Preset Action Buttons
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("📌 Load Legitimate Preset", use_container_width=True):
            st.session_state["form_values"] = dict(LEGITIMATE_SAMPLE)
            st.rerun()
    with col_p2:
        if st.button("⚠️ Load Fraud Preset", use_container_width=True):
            st.session_state["form_values"] = dict(FRAUD_SAMPLE)
            st.rerun()
    with col_p3:
        if st.button("🔄 Reset Inputs", use_container_width=True):
            st.session_state["form_values"] = dict(LEGITIMATE_SAMPLE)
            st.session_state["form_values"]["transaction_id"] = "tx-custom-001"
            st.session_state["form_values"]["Amount"] = 0.0
            st.rerun()

    current = st.session_state["form_values"]

    with st.form("transaction_submission_form"):
        # Top Metadata Row
        r1_col1, r1_col2, r1_col3 = st.columns([2, 1, 1])
        with r1_col1:
            tx_id = st.text_input(
                "Transaction ID",
                value=str(current.get("transaction_id", "tx-test-001")),
                help="Client correlation identifier for end-to-end tracing"
            )
        with r1_col2:
            time_val = st.number_input(
                "Time (seconds)",
                min_value=0.0,
                value=float(current.get("Time", 0.0)),
                step=10.0,
                help="Elapsed seconds since dataset baseline"
            )
        with r1_col3:
            amount_val = st.number_input(
                "Amount",
                min_value=0.0,
                value=float(current.get("Amount", 50.0)),
                step=5.0,
                format="%.2f",
                help="Transaction monetary amount"
            )

        # PCA Components Grid
        st.markdown("**Principal Components ($V_1$ to $V_{28}$)**")
        pca_values = {}
        cols_per_row = 4
        
        for i in range(1, 29):
            col_idx = (i - 1) % cols_per_row
            if col_idx == 0:
                row_cols = st.columns(cols_per_row)
            
            feat_name = f"V{i}"
            default_val = float(current.get(feat_name, 0.0))
            with row_cols[col_idx]:
                pca_values[feat_name] = st.number_input(
                    feat_name,
                    value=default_val,
                    format="%.4f",
                    key=f"input_{feat_name}"
                )

        submitted = st.form_submit_button("🔍 Check Transaction", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "transaction_id": tx_id if tx_id else None,
            "Time": float(time_val),
            "Amount": float(amount_val),
            **pca_values,
        }
        return True, payload

    return False, None
