"""BLL decision surface for Streamlit Community Cloud.

Self-contained: reads the bundled model artefact and a sample of scored orders,
so it needs no raw data at run time.

Community Cloud allows roughly 1 GB of memory and SHAP is the expensive step,
so explanations are computed for one selected order on demand rather than for
the whole sample up front.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

from olist import bll  # noqa: E402
from olist.bll.brief import BusinessCostModel  # noqa: E402

st.set_page_config(page_title="Delivery Risk · BLL", page_icon="📦", layout="wide")


def _find(*names: str) -> Path:
    """Locate a bundled file, tolerating a couple of layouts."""
    for name in names:
        for candidate in (HERE / name, HERE / "deploy" / name, HERE / "models" / name):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not find any of {names} near {HERE}")


@st.cache_resource
def load_model():
    path = _find("model_post_dispatch.joblib", "model_pre_dispatch.joblib")
    return joblib.load(path)


@st.cache_data
def load_data():
    orders = pd.read_parquet(_find("sample_orders.parquet"))
    background = pd.read_parquet(_find("background.parquet"))
    return orders, background


try:
    artefact = load_model()
    orders, background = load_data()
except FileNotFoundError as exc:
    st.error(f"Missing bundled files: {exc}")
    st.stop()

model = artefact["pipeline"]
cols = artefact["feature_columns"]
threshold = float(artefact["threshold"])


@st.cache_data
def score(_model, frame: pd.DataFrame, columns: tuple) -> pd.Series:
    return pd.Series(_model.predict_proba(frame[list(columns)])[:, 1], index=frame.index)


risk = score(model, orders, tuple(cols))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Cost assumptions")
    st.caption(
        "These are **assumptions**, not measured Olist finances. Every monetary "
        "figure in this brief depends on them, so they are exposed rather than "
        "buried in the code."
    )
    currency = st.text_input("Currency", "BRL")
    refund = st.slider("Refund exposure (share of order value)", 0.0, 1.0, 0.42, 0.01)
    churn = st.slider("Repeat-purchase risk (share of order value)", 0.0, 1.0, 0.46, 0.01)
    support = st.number_input("Support cost per contact", 0.0, 500.0, 12.0)
    action = st.number_input("Cost of intervening", 0.0, 500.0, 4.0)

    st.divider()
    st.caption(
        f"Model: {artefact.get('model_name', 'n/a')} · {artefact.get('scenario', 'n/a')}  \n"
        f"Decision threshold: {threshold:.3f}  \n"
        f"Trained: {artefact.get('trained_at', 'n/a')[:10]}"
    )

costs = BusinessCostModel(currency=currency, refund_share=refund, churn_share=churn,
                          support_cost=support, intervention_cost=action)

# ---------------------------------------------------------------- header
st.title("📦 Delivery risk assessment")
st.caption(
    "A late-delivery prediction re-expressed as a business decision, via the "
    "Business Language Layer (Oyeleke et al., 2026)."
)

c1, c2, c3 = st.columns(3)
c1.metric("Orders in sample", f"{len(orders):,}")
c2.metric("Risk range", f"{risk.min():.0%} – {risk.max():.0%}")
c3.metric("Flagged at threshold", f"{(risk >= threshold).mean():.0%}")

# ---------------------------------------------------------------- selection
st.divider()
mode = st.radio("Choose an order", ["Highest risk", "Pick from list"], horizontal=True)

if mode == "Highest risk":
    pos = int(risk.to_numpy().argmax())
else:
    labels = [
        f"{orders['order_id'].iloc[i][:12]}…  ·  risk {risk.iloc[i]:.0%}  ·  "
        f"{currency} {orders['total_price'].iloc[i]:,.0f}"
        for i in range(len(orders))
    ]
    pos = labels.index(st.selectbox("Order", labels))

row = orders[cols].iloc[[pos]]
order_value = float(orders["total_price"].iloc[pos])
order_id = orders["order_id"].iloc[pos]

with st.spinner("Computing SHAP explanation…"):
    explanation = bll.explain_order(model, row, background=background)
    brief = bll.build_brief(
        model, row, explanation, order_value=order_value, costs=costs,
        model_precision=artefact.get("metrics", {}).get("precision_at_chosen"),
        order_id=order_id,
    )

# ------------------------------------------------------- 1. interpret
left, right = st.columns([1, 1])
left.metric("Late-delivery probability", f"{brief.probability:.1%}", brief.risk_level)
right.success(f"**Recommended action**\n\n### {brief.recommendation.name}\n\n"
              f"{brief.recommendation.description}")

if brief.probability >= threshold:
    st.warning("Above the action threshold — intervention recommended.")
else:
    st.info("Below the action threshold — no intervention needed.")

# ---------------------------------------------------- 2. contextualise
st.subheader("Key contributing factors")
st.caption("The model's main reasons, in plain terms.")
for d in brief.drivers:
    box = st.error if d.raises_risk else st.success
    tag = "  ·  `Actionable`" if d.is_lever else ""
    box(f"**{d.label}** — {d.sentence}  \n`{d.badge}`{tag}")

st.subheader("What this means")
for line in brief.consequences:
    st.write("•", line)

# --------------------------------------------------------- 3. quantify
st.subheader("Business impact if late")
for col, (name, amount) in zip(st.columns(len(brief.impact)), brief.impact.items()):
    col.metric(name, costs.format(amount))
st.caption(f"Total exposure on this order: **{costs.format(brief.exposure)}**")

# -------------------------------------------------------- 4. recommend
st.subheader("Options")
st.dataframe(
    pd.DataFrame([
        {
            "Option": o.name,
            "Risk": f"{o.risk:.1%}" if o.feasible else "unavailable",
            "Action cost": costs.format(o.action_cost) if o.feasible else "—",
            "Expected loss": costs.format(o.expected_loss) if o.feasible else "—",
            "Recommended": "✓" if o.recommended else "",
        }
        for o in brief.options
    ]),
    hide_index=True, width="stretch",
)
st.info(f"**{brief.headline()}**  ·  Net saving vs shipping as-is: "
        f"**{costs.format(brief.net_saving)}**")

# ------------------------------------------------------------- what-if
levers = [d.feature for d in brief.drivers if d.is_lever and d.feature in cols]
if levers:
    st.subheader("What if…")
    st.caption("Counterfactuals run the full deployed pipeline, so these are the system's own figures.")
    lever = st.selectbox("Change", levers,
                         format_func=lambda f: bll.glossary.GLOSSARY[f].label)
    current = float(row[lever].iloc[0])
    new = st.slider(f"{bll.glossary.GLOSSARY[lever].label} (currently {current:.1f})",
                    0.0, float(max(current * 3, 10.0)), current, 0.5)
    new_risk = bll.counterfactual(model, row, lever, new)
    a, b = st.columns(2)
    a.metric("Risk now", f"{brief.probability:.1%}")
    b.metric("Risk after change", f"{new_risk:.1%}", f"{new_risk - brief.probability:+.1%}")

# ------------------------------------------------------- traceability
with st.expander("Audit trail — every statement and the attribution behind it"):
    st.dataframe(brief.audit(), hide_index=True, width="stretch")
    st.caption(f"Glossary coverage of shown drivers: {brief.glossary_coverage:.0%}")

st.divider()
for caveat in brief.caveats:
    st.caption(f"⚠️ {caveat}")
