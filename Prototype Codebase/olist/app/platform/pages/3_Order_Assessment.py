"""Full explainable assessment for a single order."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Order Assessment", page_icon="🔍", layout="wide")

service = st.session_state.get("service")
if service is None:
    st.warning("Open the **Home** page first to load the model.")
    st.stop()

currency = st.session_state.get("currency", "BRL")
history = service.history
risk_all = st.session_state["risk"]

# --- order selection ------------------------------------------------------
default = st.session_state.get("selected_order", int(np.argmax(risk_all)))
mode = st.radio("Choose an order", ["Highest risk", "Selected", "Pick from list"],
                index=1 if "selected_order" in st.session_state else 0,
                horizontal=True)

if mode == "Highest risk":
    pos = int(np.argmax(risk_all))
elif mode == "Selected":
    pos = default
else:
    top = np.argsort(risk_all)[::-1][:200]
    pos = st.selectbox(
        "Order", list(top),
        format_func=lambda i: f"{history['order_id'].iloc[i][:16]} · risk {risk_all[i]:.0%}",
    )

row = history.iloc[[pos]]
order_id = history["order_id"].iloc[pos]
order_value = float(history["total_price"].iloc[pos])

with st.spinner("Computing assessment…"):
    a = service.assess(row, order_value=order_value, order_id=order_id)

brief, costs = a.brief, a.brief.costs

# --- 1. header ------------------------------------------------------------
st.title("Delivery Risk Assessment")
st.caption(f"Order {order_id}")

colour = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[a.risk_level]
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Late-delivery probability", f"{a.probability:.1%}")
c1.markdown(f":{colour}[**{a.risk_level} RISK**]")
c2.metric("Model confidence", f"{a.confidence.score:.0%}")
c2.markdown(f"**{a.confidence.level} CONFIDENCE** *(estimate)*")
c3.success(f"**Recommended action**\n\n### {brief.recommendation.name}\n\n"
           f"{brief.recommendation.description}")

for warning in a.warnings:
    st.warning(warning)

if a.confidence.is_low:
    with st.expander("Why is this prediction uncertain?", expanded=True):
        for reason in a.confidence.reasons:
            st.write("•", reason)
        st.caption(
            f"Components — threshold margin {a.confidence.threshold_margin:.0%}, "
            f"input familiarity {a.confidence.familiarity:.0%}, "
            f"evidence density {a.confidence.evidence_density:.0%}."
        )

# --- 2. drivers -----------------------------------------------------------
st.divider()
st.subheader("Why this prediction?")
st.caption("The model's main reasons, in plain terms.")

for d in brief.drivers:
    box = st.error if d.raises_risk else st.success
    box(f"**{d.label}** — {d.sentence}  \n`{d.badge}`"
        + ("  ·  `Actionable`" if d.is_lever else ""))

from olist.bll import glossary

contributions = a.explanation.top_drivers(10)
chart = (pd.DataFrame({"contribution": contributions})
           .assign(factor=[glossary.describe(f, True)[0] for f in contributions.index])
           .set_index("factor"))
st.bar_chart(chart, horizontal=True)
st.caption(
    "Attributions to the model's internal risk score (log-odds). They rank the "
    "drivers against one another; they are not percentage points of the "
    "probability shown above."
)

# --- 3. similar orders ----------------------------------------------------
st.divider()
st.subheader("Similar historical orders")
similar = a.similar

if similar.n_found and similar.sufficient:
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean similarity", f"{similar.mean_similarity:.0%}")
    s2.metric("Late among similar", f"{similar.late_rate:.0%}")
    s3.metric("Mean delay when late", f"{similar.mean_delay_days:.1f} days")

    table = pd.DataFrame({
        "Order ID": similar.orders["order_id"],
        "Similarity": similar.orders["similarity"],
        "Outcome": np.where(similar.orders["is_late_delivery"] == 1, "Late", "On time"),
        "Delay (days)": similar.orders["delivery_delay_days"].round(1),
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Similarity is computed only on features knowable at prediction time. "
        "Matching on the outcome would make these statistics circular."
    )

    with st.expander("Compare with the most similar order"):
        st.dataframe(service.compare_to_similar(row[service.feature_columns], similar).round(2),
                     hide_index=True, width="stretch")
else:
    st.info(similar.note or "No sufficiently similar historical orders were found.")

# --- 4. what-if -----------------------------------------------------------
st.divider()
st.subheader("What-if analysis")

levers = [d.feature for d in brief.drivers if d.is_lever and d.feature in service.feature_columns]
levers = levers or [f for f in ("carrier_pickup_days", "estimated_delivery_days")
                    if f in service.feature_columns]

if levers:
    lever = st.selectbox("Change", levers,
                         format_func=lambda f: glossary.GLOSSARY[f].label)
    current = float(row[lever].iloc[0])
    new = st.slider(f"{glossary.GLOSSARY[lever].label} (currently {current:.1f})",
                    0.0, float(max(current * 3, 10.0)), current, 0.5)
    new_risk = service.run_counterfactual(row, lever, new)

    w1, w2, w3 = st.columns(3)
    w1.metric("Current risk", f"{a.probability:.1%}")
    w2.metric("New risk", f"{new_risk:.1%}", f"{new_risk - a.probability:+.1%}")
    w3.metric("Expected loss change",
              f"{currency} {(new_risk - a.probability) * brief.exposure:+,.2f}")

unavailable = {k: v for k, v in service.available_levers().items() if not v.available}
if unavailable:
    with st.expander("Levers this dataset cannot support"):
        for name, capability in unavailable.items():
            st.write(f"• **{name}** — {capability.reason}")
        st.caption(
            "These controls are disabled rather than simulated. Offering them would "
            "require inventing both the input and the resulting risk figure."
        )

# --- 5. counterfactuals ---------------------------------------------------
st.divider()
st.subheader("How to reduce risk")
feasible = [i for i in a.interventions if i.feasible]
if feasible:
    st.dataframe(
        pd.DataFrame([{
            "Intervention": i.name,
            "Change": f"{i.current_value:.1f} → {i.new_value:.1f}",
            "New risk": f"{i.new_risk:.1%}",
            "Risk change": f"{i.risk_change:+.1%}",
        } for i in feasible]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Each intervention is evaluated independently by re-running the model with "
        "that single input changed. These are model-estimated risk changes under a "
        "ceteris-paribus assumption, not causal guarantees, and they do not compose."
    )

# --- 6. business impact ---------------------------------------------------
st.divider()
st.subheader("Business impact")
for col, (name, amount) in zip(st.columns(len(brief.impact)), brief.impact.items()):
    col.metric(name, costs.format(amount))

b1, b2, b3 = st.columns(3)
b1.metric("Expected loss", costs.format(a.probability * brief.exposure))
b2.metric("Intervention cost", costs.format(brief.recommendation.action_cost))
b3.metric("Net saving", costs.format(brief.net_saving))

if brief.net_saving > 0:
    st.success(f"**Intervention recommended.** Escalating this order is expected to "
               f"save approximately {costs.format(brief.net_saving)}.")
else:
    st.info("**Standard fulfilment.** Intervention is not currently expected to be "
            "financially worthwhile at these cost assumptions.")

st.dataframe(
    pd.DataFrame([{
        "Option": o.name,
        "Risk": f"{o.risk:.1%}" if o.feasible else "unavailable",
        "Action cost": costs.format(o.action_cost) if o.feasible else "—",
        "Expected loss": costs.format(o.expected_loss) if o.feasible else "—",
        "Recommended": "✓" if o.recommended else "",
    } for o in brief.options]),
    hide_index=True, width="stretch",
)

# --- 7. audit and report --------------------------------------------------
st.divider()
with st.expander("Audit trail — every statement and the attribution behind it"):
    st.dataframe(brief.audit(), hide_index=True, width="stretch")
    st.caption(f"Glossary coverage of shown drivers: {brief.glossary_coverage:.0%}")

if st.button("Generate PDF report", type="primary"):
    from olist.bll import build_report

    context = {
        "Order ID": order_id,
        "Customer state": history["customer_state"].iloc[pos],
        "Seller state": history.get("seller_state", pd.Series(["—"])).iloc[pos],
        "Product category": history["product_category"].iloc[pos],
        "Purchased": f"{history['order_purchase_timestamp'].iloc[pos]:%Y-%m-%d}",
        "Promised delivery": f"{history['order_estimated_delivery_date'].iloc[pos]:%Y-%m-%d}",
        "Order value": f"{currency} {order_value:,.2f}",
    }
    with st.spinner("Building report…"):
        path = Path(tempfile.gettempdir()) / f"assessment_{str(order_id)[:12]}.pdf"
        build_report(a, path, order_context=context, model_info=service.model_info())
    st.download_button("Download report", path.read_bytes(),
                       file_name=path.name, mime="application/pdf")

for caveat in brief.caveats:
    st.caption(f"⚠️ {caveat}")
