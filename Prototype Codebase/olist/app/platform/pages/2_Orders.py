"""Order-level risk table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Orders", page_icon="📋", layout="wide")

service = st.session_state.get("service")
if service is None:
    st.warning("Open the **Home** page first to load the model.")
    st.stop()

currency = st.session_state.get("currency", "BRL")
history = service.history.copy()
history["risk"] = st.session_state["risk"]
history["risk_level"] = [service.risk_level(p) for p in history["risk"]]

st.title("Orders")
st.caption("Select an order to open its full assessment on the Order Assessment page.")

c1, c2, c3 = st.columns([2, 1, 1])
level = c1.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH"])
sort_by = c2.selectbox("Sort by", ["Risk (high to low)", "Expected loss", "Order value"])
limit = c3.number_input("Rows", 10, 500, 50, 10)

view = history[history["risk_level"].isin(level)] if level else history
costs = service.costs
view = view.assign(
    exposure=view["total_price"] * (costs.refund_share + costs.churn_share) + costs.support_cost
)
view["expected_loss"] = view["risk"] * view["exposure"]

view = view.sort_values(
    {"Risk (high to low)": "risk", "Expected loss": "expected_loss",
     "Order value": "total_price"}[sort_by],
    ascending=False,
).head(int(limit))

table = pd.DataFrame({
    "Order ID": view["order_id"],
    "State": view.get("customer_state"),
    "Category": view.get("product_category"),
    "Purchased": view["order_purchase_timestamp"].dt.date,
    "Risk": view["risk"],
    "Level": view["risk_level"],
    f"Value ({currency})": view["total_price"].round(2),
    f"Expected loss ({currency})": view["expected_loss"].round(2),
    "Action": ["Escalate" if r >= service.threshold else "Standard" for r in view["risk"]],
})

st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={"Risk": st.column_config.ProgressColumn(
        "Risk", format="%.1f%%", min_value=0.0, max_value=1.0)},
)

st.divider()
chosen = st.selectbox(
    "Open an order",
    view.index,
    format_func=lambda i: f"{view.loc[i, 'order_id'][:16]} · risk {view.loc[i, 'risk']:.0%}",
)
if st.button("Open assessment", type="primary"):
    st.session_state["selected_order"] = int(chosen)
    st.switch_page("pages/3_Order_Assessment.py")
