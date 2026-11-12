"""Global model behaviour, model card, and capability disclosure."""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")

service = st.session_state.get("service")
if service is None:
    st.warning("Open the **Home** page first to load the model.")
    st.stop()

st.title("Model Insights")

st.subheader("Model card")
st.dataframe(
    pd.DataFrame(service.model_info().items(), columns=["Field", "Value"]),
    hide_index=True, width="stretch",
)
st.caption("Only metrics actually recorded in the trained artefact are shown.")

st.divider()
st.subheader("Overall feature importance")
st.caption(
    "Mean absolute SHAP contribution across a sample of historical orders. "
    "Permutation and SHAP importance are preferred over impurity-based importance, "
    "which is biased toward high-cardinality continuous features."
)

with st.spinner("Computing global importance…"):
    importance = service.get_feature_importance()

if importance.empty:
    st.info("No historical orders are loaded, so global importance cannot be computed.")
else:
    top = importance.head(15)
    st.bar_chart(top.set_index("label")["share"], horizontal=True)
    st.dataframe(
        top[["label", "feature", "share"]].assign(share=lambda d: (d["share"] * 100).round(1))
           .rename(columns={"label": "Factor", "feature": "Feature", "share": "Share (%)"}),
        hide_index=True, width="stretch",
    )

st.divider()
st.subheader("What this dataset supports")
st.caption(
    "Capabilities are disclosed rather than silently omitted. Where a lever is "
    "unavailable, the field that would be required is named."
)

levers = service.available_levers()
st.dataframe(
    pd.DataFrame([{
        "Capability": name,
        "Available": "yes" if cap.available else "no",
        "Reason / requirement": cap.reason or "—",
    } for name, cap in levers.items()]),
    hide_index=True, width="stretch",
)

st.info(
    "**On the scenario distinction.** A `post_dispatch` model uses carrier handover "
    "timing and can only run once the parcel has left the seller. A `pre_dispatch` "
    "model runs at order time but is substantially weaker. Which one is loaded "
    "determines which interventions are possible."
)
