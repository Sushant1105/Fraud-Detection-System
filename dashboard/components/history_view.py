"""
Transaction history and audit analytics component for the Streamlit dashboard.
Displays persistent transaction logs, database telemetry, and transaction inspector.
Strictly decoupled: Communicates with FastAPI via FraudApiClient only.
"""
import streamlit as st
import pandas as pd
from typing import Optional
from dashboard.services.api_client import FraudApiClient
from dashboard.components.shap_charts import render_shap_explanations


def render_history_view(client: FraudApiClient):
    """
    Renders the persistent transaction history, database statistics,
    and single-transaction SHAP audit inspector.
    """
    st.header("🗄️ Database Audit & Transaction History")
    st.write(
        "Persistent audit records stored in SQLite and queried via FastAPI REST endpoints. "
        "Each transaction record retains exact model outputs, risk tier classifications, and SHAP attributions."
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    # 1. Summary Statistics from GET /stats
    stats_success, stats_data, stats_err = client.get_stats()

    if not stats_success or not stats_data:
        st.warning(f"⚠️ Unable to retrieve statistics from API: {stats_err or 'No data'}")
    else:
        st.subheader("📊 Aggregate Database Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Transactions", f"{stats_data.get('total_transactions', 0):,}")
        m2.metric(
            "Fraud Detected",
            f"{stats_data.get('fraud_count', 0):,}",
            delta=f"{stats_data.get('fraud_rate', 0.0):.1f}% fraud rate",
            delta_color="inverse",
        )
        m3.metric("Legitimate Cleared", f"{stats_data.get('legitimate_count', 0):,}")
        m4.metric("Fraud Rate", f"{stats_data.get('fraud_rate', 0.0):.2f}%")

    st.write("")

    # 2. Transaction History Table from GET /transactions
    st.subheader("📋 Recent Transaction History")
    tx_success, tx_data, tx_err = client.get_transactions(limit=50)

    if not tx_success or not tx_data:
        st.error(f"❌ Failed to fetch transactions from API: {tx_err or 'Unknown error'}")
        return

    transactions = tx_data.get("transactions", [])
    if not transactions:
        st.info(
            "ℹ️ No transactions have been recorded in the database yet.\n\n"
            "Submit a transaction using the **'⚡ Real-Time Scoring'** tab to create persistent audit records."
        )
        return

    # Build DataFrame for presentation
    df_history = pd.DataFrame(transactions)
    table_df = pd.DataFrame({
        "Database ID": df_history["id"],
        "Transaction ID": df_history["transaction_id"].fillna("(None)"),
        "Amount": df_history["amount"].apply(lambda a: f"€{a:,.2f}"),
        "Fraud Probability": df_history["fraud_probability"].apply(lambda p: f"{p:.4f}"),
        "Threshold": df_history["threshold"].apply(lambda t: f"{t:.4f}"),
        "Decision": df_history["decision"].apply(
            lambda d: "🔴 FRAUD" if d == "FRAUD" else "🟢 LEGITIMATE"
        ),
        "Risk Level": df_history["risk_level"],
        "Timestamp (UTC)": df_history["created_at"],
    })

    st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing latest {len(transactions)} of {tx_data.get('total', len(transactions))} recorded transactions.")

    st.divider()

    # 3. Transaction Details & Stored SHAP Inspector
    st.subheader("🔍 Transaction Detail & SHAP Explanation Inspector")
    st.write(
        "Look up any transaction by its client `transaction_id` via `GET /transactions/{transaction_id}` "
        "to inspect its stored scoring details and exact SHAP feature attributions."
    )

    # Options with client transaction_id
    valid_tx_ids = [
        t["transaction_id"] for t in transactions if t.get("transaction_id")
    ]

    selected_tx_id: Optional[str] = None
    col_sel, col_manual = st.columns([1, 1])

    with col_sel:
        if valid_tx_ids:
            # Provide selectbox from recent transactions
            chosen = st.selectbox(
                "Select from recent transaction IDs:",
                options=[""] + valid_tx_ids,
                index=0,
            )
            if chosen:
                selected_tx_id = chosen

    with col_manual:
        manual_id = st.text_input(
            "Or enter specific transaction ID:",
            value="",
            placeholder="e.g. tx-fraud-sample",
        )
        if manual_id.strip():
            selected_tx_id = manual_id.strip()

    if selected_tx_id:
        with st.spinner(f"Querying transaction '{selected_tx_id}' via API..."):
            single_success, single_record, single_err = client.get_transaction(selected_tx_id)

        if not single_success or not single_record:
            st.error(f"❌ {single_err or 'Transaction not found'}")
        else:
            # Display record summary in cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transaction ID", str(single_record.get("transaction_id", "(None)")))
            c2.metric("Amount", f"€{single_record.get('amount', 0.0):,.2f}")
            c3.metric(
                "Fraud Probability",
                f"{single_record.get('fraud_probability', 0.0):.4f}",
                delta=f"Threshold: {single_record.get('threshold', 0.0):.4f}",
            )
            c4.metric(
                "Decision",
                str(single_record.get("decision", "")),
                delta=f"Risk: {single_record.get('risk_level', '')}",
            )

            # Metadata expander
            with st.expander("ℹ️ Model & Persistence Metadata", expanded=False):
                st.json({
                    "database_id": single_record.get("id"),
                    "transaction_id": single_record.get("transaction_id"),
                    "time": single_record.get("time"),
                    "amount": single_record.get("amount"),
                    "model_algorithm": single_record.get("model_algorithm"),
                    "model_version": single_record.get("model_version"),
                    "latency_ms": single_record.get("latency_ms"),
                    "created_at": single_record.get("created_at"),
                })

            # Render exact stored SHAP features
            stored_features = single_record.get("top_contributing_features", [])
            render_shap_explanations(stored_features)
