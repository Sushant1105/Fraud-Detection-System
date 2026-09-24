"""
SHAP feature attribution bar chart and data table component.
Consumes top contributing features returned dynamically by the FastAPI backend.
"""
import streamlit as st
import pandas as pd
import altair as alt
from typing import List, Dict, Any


def render_shap_explanations(top_features: List[Dict[str, Any]]):
    """
    Render SHAP feature attribution bar chart and detailed data breakdown.
    Does NOT hardcode feature names; dynamically renders API response.
    """
    st.subheader("🔍 Prediction Explainability (SHAP Risk Attribution)")
    st.write(
        "Decomposition of the model's decision margin into individual feature contributions. "
        "Positive values push toward fraud; negative values push toward legitimate."
    )

    if not top_features:
        st.info("No feature contributions available.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(top_features)
    # Ensure correct column naming and formatting
    df["display_direction"] = df["direction"].apply(
        lambda d: "🔴 Increases Risk" if d == "increases_risk" else "🟢 Decreases Risk"
    )
    df["formatted_shap"] = df["shap_value"].apply(lambda v: f"{v:+.4f}")
    df["formatted_share"] = df["relative_attribution_share"].apply(lambda s: f"{s:.2f}%")

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.markdown("**Top Feature Attribution Impacts (SHAP Log-Odds Margin)**")
        
        # Horizontal Bar Chart via Altair
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("shap_value:Q", title="SHAP Value (Margin Impact)"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["increases_risk", "decreases_risk"],
                        range=["#d62728", "#2ca02c"]
                    ),
                    legend=alt.Legend(title="Risk Impact")
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("feature_value:Q", title="Raw Value", format=".4f"),
                    alt.Tooltip("shap_value:Q", title="SHAP Impact", format="+.4f"),
                    alt.Tooltip("direction:N", title="Direction"),
                    alt.Tooltip("relative_attribution_share:Q", title="Relative Share (%)", format=".2f"),
                ]
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

    with col_table:
        st.markdown("**Detailed Feature Contribution Breakdown**")
        display_table = df[[
            "feature", "feature_value", "formatted_shap", "display_direction", "formatted_share"
        ]].rename(columns={
            "feature": "Feature",
            "feature_value": "Raw Value",
            "formatted_shap": "SHAP Impact",
            "display_direction": "Risk Direction",
            "formatted_share": "Attribution Share"
        })
        st.dataframe(display_table, use_container_width=True, hide_index=True)
