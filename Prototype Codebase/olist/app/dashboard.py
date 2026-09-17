"""E-commerce Delivery Risk Prediction — Explainable AI Dashboard.

A single dense decision screen with a navigation rail, laid out as numbered
panels so an operations lead can take in one order's whole picture without
scrolling between contexts.

Every figure is computed from the deployed model. Capabilities the Olist data
cannot support — courier, warehouse, weather, seller rating — are shown as
unavailable with the missing field named, rather than being simulated.

Run with:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", Path.cwd() / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from olist.bll import PredictionService, glossary  # noqa: E402
from olist.bll.brief import BusinessCostModel  # noqa: E402

st.set_page_config(page_title="Delivery Risk Intelligence", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown("""
<style>
  :root {
    --ink:      #16212B;
    --muted:    #6B7A88;
    --hairline: #E3E8EC;
    --surface:  #F7F9FA;
    --brand:    #12352A;
    --high:     #C0392B;
    --medium:   #C77700;
    --low:      #1E7F4F;
  }
  .block-container { padding: 1.8rem 2.4rem 4rem; max-width: 1500px; }
  header[data-testid="stHeader"] { background: transparent; }

  /* Navigation rail */
  section[data-testid="stSidebar"] { background: var(--brand); width: 250px !important; }
  section[data-testid="stSidebar"] * { color: #E8EFEA !important; }
  /* Navigation items: a tap target with room around it reads as a menu;
     tightly stacked radio buttons read as a form. Hover and selection are
     distinguished by an accent rail rather than by shade alone, so the
     current page is identifiable without relying on colour discrimination. */
  section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: .42rem;
    padding-top: .3rem;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    font-size: .95rem; font-weight: 500;
    padding: .68rem .8rem .68rem .72rem;
    border-radius: 8px; width: 100%;
    border-left: 3px solid transparent;
    cursor: pointer;
    transition: background .16s ease, border-color .16s ease,
                transform .16s ease, padding-left .16s ease;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, .09);
    border-left-color: rgba(232, 163, 61, .55);
    padding-left: .95rem;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:active {
    transform: scale(.985);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:focus-visible) {
    outline: 2px solid #E8A33D;
    outline-offset: 2px;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255, 255, 255, .14);
    border-left-color: #E8A33D;
    font-weight: 650;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    margin-right: .1rem;
  }
  @media (prefers-reduced-motion: reduce) {
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
      transition: none;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:active {
      transform: none;
    }
  }
  section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
  section[data-testid="stSidebar"] .stExpander {
    border-color: rgba(255,255,255,.16) !important;
  }
  .brand { font-size: 1.28rem; font-weight: 700; letter-spacing: -.01em; }
  .brand-sub {
    font-family: ui-monospace, monospace; font-size: .62rem;
    letter-spacing: .16em; text-transform: uppercase; opacity: .65;
    margin-bottom: 1.5rem;
  }
  .rail-note { font-size: .76rem; line-height: 1.6; opacity: .62; }
  section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

  /* Cards. Streamlit's bordered container is the element that actually wraps
     each panel; this gives it breathing room and a consistent edge. */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px; border-color: var(--hairline);
    padding: 1.25rem 1.4rem; background: #fff;
  }
  div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlockBorderWrapper"]) {
    margin-bottom: .5rem;
  }
  .panel-title {
    font-size: 1.02rem; font-weight: 700; color: var(--ink);
    margin-bottom: .15rem;
  }
  .panel-sub {
    font-size: .8rem; color: var(--muted); margin-bottom: 1rem;
  }
  .panel-no {
    display: inline-block; font-family: ui-monospace, monospace;
    font-size: .68rem; font-weight: 700; color: #fff; background: var(--ink);
    border-radius: 3px; padding: .05rem .32rem; margin-right: .4rem;
  }

  .eyebrow {
    font-family: ui-monospace, monospace; font-size: .64rem;
    letter-spacing: .13em; text-transform: uppercase; color: var(--muted);
  }
  .figure {
    font-family: ui-monospace, monospace; font-size: 2.5rem; font-weight: 700;
    line-height: 1.05; font-variant-numeric: tabular-nums; color: var(--ink);
  }
  .figure-sm {
    font-family: ui-monospace, monospace; font-size: 1.25rem; font-weight: 650;
    font-variant-numeric: tabular-nums; color: var(--ink);
  }
  .pill {
    display: inline-block; font-size: .72rem; font-weight: 700;
    border-radius: 20px; padding: .18rem .7rem;
  }
  .pill-high   { background: #FBEAE8; color: var(--high); }
  .pill-medium { background: #FDF3E3; color: var(--medium); }
  .pill-low    { background: #E8F5EE; color: var(--low); }

  .factor {
    border-radius: 6px; padding: .7rem .9rem; margin-bottom: .55rem;
  }
  .factor-up   { background: #FBEAE8; }
  .factor-down { background: #E8F5EE; }
  .factor-name { font-size: .92rem; font-weight: 650; margin-bottom: .15rem; }
  .factor-up .factor-name   { color: var(--high); }
  .factor-down .factor-name { color: var(--low); }
  .factor-text { font-size: .84rem; color: var(--muted); line-height: 1.5; }

  .action { font-size: .88rem; color: var(--ink); margin-bottom: .45rem; line-height: 1.5; }
  .caption { font-size: .8rem; color: var(--muted); line-height: 1.55; }
  .unavailable {
    font-size: .8rem; line-height: 1.6; color: var(--muted); background: var(--surface);
    border: 1px dashed var(--hairline); border-radius: 6px; padding: .5rem .7rem;
  }
  .footer {
    border-top: 1px solid var(--hairline); margin-top: 1.4rem; padding-top: .7rem;
    font-size: .72rem; color: var(--muted);
  }
  div[data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

PILL = {"HIGH": "pill-high", "MEDIUM": "pill-medium", "LOW": "pill-low"}


def panel_head(number: str, title: str, sub: str = "") -> str:
    return (f'<div class="panel-title"><span class="panel-no">{number}</span>{title}</div>'
            + (f'<div class="panel-sub">{sub}</div>' if sub else ""))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

MODEL_DIRS = [ROOT / "models", ROOT / "deploy", ROOT, Path.cwd()]
DATA_DIRS = [ROOT / "data" / "processed", ROOT / "deploy", ROOT, Path.cwd()]


def _find(patterns: list[str], dirs: list[Path]) -> Path | None:
    for directory in dirs:
        if directory.exists():
            for pattern in patterns:
                hits = sorted(directory.glob(pattern))
                if hits:
                    return hits[0]
    return None


@st.cache_resource(show_spinner="Loading model…")
def load_service(refund, churn, support, action, currency):
    model_path = _find(["model_post_dispatch.joblib", "model_*.joblib"], MODEL_DIRS)
    if model_path is None:
        return None
    return PredictionService.from_files(
        model_path,
        _find(["orders_features.parquet", "sample_orders.parquet"], DATA_DIRS),
        _find(["background.parquet"], DATA_DIRS),
        costs=BusinessCostModel(currency=currency, refund_share=refund,
                                churn_share=churn, support_cost=support,
                                intervention_cost=action),
    )


# --------------------------------------------------------------------------
# Navigation rail
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="brand">📦 Business Language Layer</div>'
                '<div class="brand-sub">Delivery Risk Intelligence</div>',
                unsafe_allow_html=True)
    st.write("")

    view = st.radio(
        "Navigation",
        ["Overview", "Risk Dashboard", "Sales Analysis", "Orders",
         "Explainability", "What-If Analysis", "Model Insights"],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("Cost assumptions"):
        st.markdown('<div class="rail-note">Assumptions, not measured Olist '
                    'finances. Every figure moves with them.</div>',
                    unsafe_allow_html=True)
        currency = st.text_input("Currency", "BRL")
        refund = st.slider("Refund exposure", 0.0, 1.0, 0.42, 0.01)
        churn = st.slider("Repeat-purchase risk", 0.0, 1.0, 0.46, 0.01)
        support = st.number_input("Support cost per contact", 0.0, 500.0, 12.0)
        action = st.number_input("Cost of intervening", 0.0, 500.0, 4.0)

    st.markdown('<div class="rail-note">Explainable AI decision support.<br>'
                'Predict the risk · Explain it · Reduce it · Price the decision.</div>',
                unsafe_allow_html=True)

service = load_service(refund, churn, support, action, currency)

if service is None:
    st.error("**No trained model found.** Expected `model_post_dispatch.joblib` "
             "in `models/` or `deploy/`.")
    st.code("python -m olist.validate --build\n"
            "python -m olist.train --scenario post_dispatch", language="bash")
    st.stop()

history = service.history
if not len(history):
    st.error("No historical orders found. Run `python -m olist.validate --build`.")
    st.stop()

if "risk" not in st.session_state:
    with st.spinner("Scoring orders…"):
        st.session_state["risk"] = service.predict_probability(history)
risk_all = st.session_state["risk"]
costs = service.costs


# ==========================================================================
# Order-level views
# ==========================================================================

def select_order() -> int:
    c1, c2 = st.columns([1, 2])
    mode = c1.selectbox("Select", ["Typical high-risk", "Highest risk", "From the queue"],
                        label_visibility="collapsed")
    if mode == "Highest risk":
        return int(np.argmax(risk_all))
    if mode == "Typical high-risk":
        # The single most extreme order invites "so it is always certain?".
        # The 90th percentile is closer to a normal day's worst case.
        return int(np.abs(risk_all - np.percentile(risk_all, 90)).argmin())
    top = np.argsort(risk_all)[::-1][:150]
    return c2.selectbox("Order", list(top), label_visibility="collapsed",
                        format_func=lambda i: f"{history['order_id'].iloc[i][:16]} · {risk_all[i]:.0%}")


def assess(pos: int):
    row = history.iloc[[pos]]
    return row, service.assess(
        row, order_value=float(history["total_price"].iloc[pos]),
        order_id=history["order_id"].iloc[pos],
    )


def header_strip(pos: int, a) -> None:
    """Order identity and the two headline gauges."""
    cols = st.columns([1.4, 1, 1, 1, 1])
    o = history.iloc[pos]

    with cols[0]:
        st.markdown('<div class="eyebrow">Order ID</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="figure-sm">{str(o["order_id"])[:16]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="caption">{o["order_purchase_timestamp"]:%d %b %Y, %H:%M}</div>',
                    unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="eyebrow">Destination</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="figure-sm">{o["customer_state"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="caption">{str(o.get("customer_city", "—")).title()}</div>',
                    unsafe_allow_html=True)

    with cols[2]:
        st.markdown('<div class="eyebrow">Origin</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="figure-sm">{o.get("seller_state", "—")}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="caption">{o["product_category"][:22]}</div>',
                    unsafe_allow_html=True)

    with cols[3]:
        st.markdown('<div class="eyebrow">Predicted late-delivery risk</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="figure">{a.probability:.0%}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="pill {PILL[a.risk_level]}">{a.risk_level} RISK</span>'
                    f'<span class="caption"> threshold {service.threshold:.0%}</span>',
                    unsafe_allow_html=True)
        st.progress(float(a.probability))

    with cols[4]:
        st.markdown('<div class="eyebrow">Model confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="figure">{a.confidence.score:.0%}</div>',
                    unsafe_allow_html=True)
        near_line = a.confidence.threshold_margin < 0.25
        why = ("sits close to the action threshold, so the recommendation could flip"
               if near_line else "is well clear of the action threshold")
        st.markdown(f'<span class="caption"><b>{a.confidence.level}</b> — how much to '
                    f'trust this estimate, not how likely lateness is. This order '
                    f'{why}.</span>', unsafe_allow_html=True)
        st.progress(float(a.confidence.score))


def panel_why(a) -> None:
    st.markdown(panel_head("1", "Why this prediction?",
                           "SHAP local attribution, expressed in business terms"),
                unsafe_allow_html=True)
    for d in a.brief.drivers:
        cls = "factor-up" if d.raises_risk else "factor-down"
        st.markdown(f'<div class="factor {cls}">'
                    f'<div class="factor-name">{d.label} &nbsp; {d.shap_value:+.3f}</div>'
                    f'<div class="factor-text">{d.sentence}</div></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="caption">Red raises risk · green lowers it. Values are '
                'attributions to the model\'s internal score, so they rank the drivers '
                'rather than decomposing the percentage above.</div>',
                unsafe_allow_html=True)


def panel_importance() -> None:
    st.markdown(panel_head("2", "Overall model feature importance",
                           "Mean absolute SHAP across historical orders"),
                unsafe_allow_html=True)
    with st.spinner("Computing…"):
        imp = service.get_feature_importance()
    if imp.empty:
        st.info("No historical orders loaded.")
        return
    top = imp.head(10).set_index("label")["share"]
    st.bar_chart(top, horizontal=True, height=300)

    lead, second = imp.iloc[0], imp.iloc[1]
    top3 = imp.head(3)["share"].sum()
    st.markdown(
        f'<div class="caption"><b>How to read this.</b> Averaged across historical '
        f'orders, <b>{lead["label"]}</b> is the strongest driver at {lead["share"]:.0%} '
        f'of total attribution, ahead of <b>{second["label"]}</b> at '
        f'{second["share"]:.0%}. The top three account for {top3:.0%}, so the model '
        f'leans on a handful of signals rather than spreading thinly. This is global '
        f'behaviour; panel 1 shows what drove <i>this</i> order, which can differ.</div>',
        unsafe_allow_html=True)


def panel_similar(a) -> None:
    st.markdown(panel_head("3", "Most similar historical orders",
                           "Matched only on features knowable before delivery"),
                unsafe_allow_html=True)
    s = a.similar
    if not (s.n_found and s.sufficient):
        st.info(s.note or "No sufficiently similar historical orders were found.")
        return
    st.dataframe(
        pd.DataFrame({
            "Order": s.orders["order_id"].str[:12],
            "Similarity": s.orders["similarity"].map("{:.0%}".format),
            "Outcome": np.where(s.orders["is_late_delivery"] == 1, "Late", "On time"),
            "Delay (d)": s.orders["delivery_delay_days"].round(1),
        }).head(6),
        hide_index=True, width="stretch", height=240,
    )
    c1, c2 = st.columns(2)
    c1.metric("Mean similarity", f"{s.mean_similarity:.0%}")
    c2.metric("Late among similar", f"{s.late_rate:.0%}")


def panel_comparison(row, a) -> None:
    st.markdown(panel_head("4", "Risk comparison",
                           "This order against its closest historical match"),
                unsafe_allow_html=True)
    table = service.compare_to_similar(row[service.feature_columns], a.similar)
    if table.empty:
        st.info("No comparable order available.")
        return
    table = table.head(7).copy()
    table["feature"] = table["feature"].map(lambda f: glossary.describe(f, True)[0])
    st.dataframe(
        table.rename(columns={"feature": "Feature", "current_order": "This order",
                              "most_similar": "Most similar", "difference": "Diff"}).round(1),
        hide_index=True, width="stretch", height=290,
    )


def panel_whatif(row, a) -> None:
    st.markdown(panel_head("5", "What-if analysis",
                           "Adjust a lever and the model re-scores the order"),
                unsafe_allow_html=True)

    levers = [f for f in service.feature_columns if glossary.is_lever(f)]
    if levers:
        lever = st.selectbox("Lever", levers,
                             format_func=lambda f: glossary.GLOSSARY[f].label)
        current = float(row[lever].iloc[0])
        new = st.slider(f"{glossary.GLOSSARY[lever].label} (now {current:.1f})",
                        0.0, float(max(current * 3, 10.0)), current, 0.5)
        new_risk = service.run_counterfactual(row, lever, new)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current risk", f"{a.probability:.0%}")
        c2.metric("New risk", f"{new_risk:.0%}", f"{new_risk - a.probability:+.0%}")
        c3.metric("Expected loss change",
                  f"{(new_risk - a.probability) * a.brief.exposure:+,.0f}")

    unavailable = {k: v for k, v in service.available_levers().items() if not v.available}
    if unavailable:
        rows = "".join(f"· <b>{n}</b> — {c.reason}<br>" for n, c in unavailable.items())
        st.markdown(f'<div class="unavailable"><b>Levers this dataset cannot support</b><br>'
                    f'{rows}Disabled rather than simulated: offering them would mean '
                    f'inventing both the control and the resulting figure.</div>',
                    unsafe_allow_html=True)


def panel_actions(a) -> None:
    st.markdown(panel_head("6", "Recommended actions",
                           "Ranked by expected cost, priced against doing nothing"),
                unsafe_allow_html=True)
    brief = a.brief
    st.markdown(f"**{brief.recommendation.name}** — {brief.recommendation.description}")
    st.write("")
    for o in brief.options:
        if not o.feasible:
            continue
        mark = "✓" if o.recommended else "·"
        st.markdown(f'<div class="action">{mark} <b>{o.name}</b> — risk {o.risk:.0%}, '
                    f'cost {costs.format(o.action_cost)}, expected loss '
                    f'{costs.format(o.expected_loss)}</div>', unsafe_allow_html=True)
    st.write("")
    if brief.net_saving > 0:
        st.success(f"Acting is expected to save **{costs.format(brief.net_saving)}** "
                   f"on this order.")
    else:
        st.info("Standard fulfilment — intervention is not expected to pay at these "
                "cost assumptions.")


def panel_outcomes(a) -> None:
    st.markdown(panel_head("7", "Similar past outcomes",
                           "What actually happened to comparable orders"),
                unsafe_allow_html=True)
    s = a.similar
    if not (s.n_found and s.sufficient):
        st.info(s.note or "Not enough comparable orders.")
        return
    late = int(s.orders["is_late_delivery"].sum())
    st.markdown(f'<div class="figure">{late}/{s.n_found}</div>'
                f'<div class="caption">of the most similar orders arrived late</div>',
                unsafe_allow_html=True)
    st.progress(late / s.n_found)
    st.metric("Mean delay when late", f"{s.mean_delay_days:.1f} days")
    st.markdown('<div class="caption">Independent of the model: these are observed '
                'outcomes, not predictions.</div>', unsafe_allow_html=True)


def panel_counterfactual(row, a) -> None:
    st.markdown(panel_head("8", "How to reduce risk",
                           "What would each lever have to reach for a lower band?"),
                unsafe_allow_html=True)
    target = 0.20

    # If the order is already inside the target band there is nothing to
    # reduce, and listing levers would imply a change is needed.
    if a.probability <= target + 0.002:
        st.markdown(
            f'<div class="action">This order is already at '
            f'<b>{a.probability:.0%}</b>, within the {target:.0%} low-risk band. '
            f'No lever needs to move.</div>', unsafe_allow_html=True)
        st.markdown('<div class="caption">The question this panel answers — what '
                    'would it take to get lower — only arises once an order is '
                    'above the band.</div>', unsafe_allow_html=True)
        return

    found_any = False
    for feature in [f for f in service.feature_columns if glossary.is_lever(f)][:4]:
        label = glossary.GLOSSARY[feature].label
        r = service.find_value_for_target(row, feature, target)
        if r.already_satisfied:
            continue
        if r.achievable:
            found_any = True
            st.markdown(f'<div class="action">↓ <b>{label}</b> — '
                        f'{r.current_value:.1f} → <b>{r.required_value:.1f}</b> '
                        f'gives {r.achieved_risk:.0%}</div>', unsafe_allow_html=True)
        elif r.best_risk is not None:
            st.markdown(f'<div class="action">✕ <b>{label}</b> — cannot reach '
                        f'{target:.0%} alone (best {r.best_risk:.0%})</div>',
                        unsafe_allow_html=True)
    if not found_any:
        st.markdown('<div class="caption">No single lever brings this order into the '
                    'low-risk band. That is a finding, not a gap: it means the risk is '
                    'structural rather than fixable by one adjustment.</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="caption">Each change is evaluated independently. These are '
                'model-estimated associations under a ceteris-paribus assumption, not '
                'causal guarantees, and they do not compose.</div>',
                unsafe_allow_html=True)


def _kpi_box(col, label, value, fg, bg, subtitle=""):
    """Render one metric in a coloured card."""
    sub = f'<div style="font-size:.75rem;color:{fg};margin-top:.18rem">{subtitle}</div>' if subtitle else ""
    col.markdown(
        f'''<div style="background:{bg};border-radius:10px;padding:.9rem 1rem;
            border-left:4px solid {fg};height:100%">
        <div style="font-size:.74rem;color:#6B7A88;font-weight:600;
            letter-spacing:.06em;text-transform:uppercase;margin-bottom:.28rem">
        {label}</div>
        <div style="font-size:1.45rem;font-weight:700;color:{fg};
            font-variant-numeric:tabular-nums;line-height:1.1">
        {value}</div>{sub}</div>''',
        unsafe_allow_html=True,
    )


def panel_impact(a) -> None:
    brief = a.brief
    st.markdown(panel_head("9", "Business impact",
                           "What the risk costs, and whether acting pays"),
                unsafe_allow_html=True)

    # ── Row 1: cost components — amber, they are the exposure at stake ───────
    st.markdown(
        '<div style="font-size:.8rem;color:#6B7A88;margin-bottom:.5rem">' +
        "What a late delivery costs on this order</div>",
        unsafe_allow_html=True)
    cols = st.columns(len(brief.impact))
    for col, (name, amount) in zip(cols, brief.impact.items()):
        _kpi_box(col, name, costs.format(amount), "#C77700", "#FDF3E3")

    st.write("")

    # ── Row 2: decision numbers — colours carry meaning ──────────────────────
    # Red   = money at risk (expected loss without acting)
    # Blue  = cost to intervene (controllable spend)
    # Green = net saving (the case for acting)
    st.markdown(
        '<div style="font-size:.8rem;color:#6B7A88;margin-bottom:.5rem">' +
        "Whether acting is worth it</div>",
        unsafe_allow_html=True)
    expected_loss = a.probability * brief.exposure
    c1, c2, c3 = st.columns(3)
    _kpi_box(c1, "Expected loss if not acted",
             costs.format(expected_loss), "#B3261E", "#FBEAE8",
             subtitle=f"{a.probability:.0%} × {costs.format(brief.exposure)} exposure")
    _kpi_box(c2, "Cost to act",
             costs.format(brief.recommendation.action_cost),
             "#1A56DB", "#EBF5FF",
             subtitle=brief.recommendation.name)
    net_fg = "#1E7F4F" if brief.net_saving > 0 else "#B3261E"
    net_bg = "#E8F5EE" if brief.net_saving > 0 else "#FBEAE8"
    _kpi_box(c3, "Net saving",
             costs.format(brief.net_saving),
             net_fg, net_bg,
             subtitle="Acting beats doing nothing" if brief.net_saving > 0
                      else "Intervention does not pay at these assumptions")

    st.write("")

    best_alt = min((o for o in brief.options if o.feasible and not o.recommended),
                   key=lambda o: o.total_cost, default=None)
    if brief.net_saving > 0 and best_alt is not None:
        st.info(
            f"In plain terms: there is a **{a.probability:.0%}** chance this "
            f"{costs.format(a.brief.order_value)} order is lost to a late delivery — "
            f"on average **{costs.format(expected_loss)}**. "
            f"{brief.recommendation.name} costs "
            f"{costs.format(brief.recommendation.action_cost)} and lowers the risk to "
            f"**{brief.recommendation.risk:.0%}**, so acting now saves about "
            f"**{costs.format(brief.net_saving)}**."
        )


# ==========================================================================
# Views
# ==========================================================================

# ── Overview hero ──────────────────────────────────────────────────────────
# The graphic is a route line: purchase, transit, the promised date, and the
# zone beyond it where an order is late. That is what the whole application
# is about — a parcel racing a promise — so it earns the space rather than
# decorating it. It draws once on load, left to right, and holds still after.
if view == "Overview":
    late_rate = (history["is_late_delivery"].mean()
                 if "is_late_delivery" in history.columns else None)
    auc = service.artefact.get("metrics", {}).get("roc_auc")

    facts = [(f"{len(history):,}", "orders scored")]
    if late_rate is not None:
        facts.append((f"{late_rate:.2%}", "arrived late"))
    if auc:
        facts.append((f"{auc:.3f}", "model AUC"))
    facts.append((f"{service.threshold:.0%}", "action threshold"))

    fact_html = "".join(
        f'<div class="fact"><span class="fact-n">{n}</span>'
        f'<span class="fact-l">{l}</span></div>'
        for n, l in facts
    )

    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&display=swap');

  .hero {{
    position: relative;
    background: linear-gradient(160deg, #0F2A22 0%, #14392E 55%, #10302791 100%);
    border-radius: 14px;
    padding: 2.1rem 2.3rem 1.7rem;
    margin-bottom: 1.4rem;
    overflow: hidden;
  }}
  .hero::after {{            /* faint cartographic grid, barely there */
    content: "";
    position: absolute; inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
    background-size: 34px 34px;
    pointer-events: none;
  }}
  .hero > * {{ position: relative; z-index: 1; }}

  .hero-live {{
    display: inline-flex; align-items: center; gap: .45rem;
    font-family: Archivo, system-ui, sans-serif;
    font-size: .74rem; font-weight: 600; color: #8FBFA8;
    margin-bottom: .9rem;
  }}
  .hero-live i {{
    width: 7px; height: 7px; border-radius: 50%;
    background: #4ADE80; display: inline-block;
    animation: breathe 2.8s ease-in-out infinite;
  }}
  @keyframes breathe {{
    0%, 100% {{ opacity: 1;   box-shadow: 0 0 0 0 rgba(74,222,128,.45); }}
    50%      {{ opacity: .55; box-shadow: 0 0 0 5px rgba(74,222,128,0); }}
  }}

  .hero h1 {{
    font-family: Archivo, system-ui, sans-serif;
    font-size: 2.55rem; font-weight: 800; line-height: 1.04;
    letter-spacing: -.028em; color: #F2F7F4;
    margin: 0 0 .45rem;
  }}
  .hero p {{
    font-family: Archivo, system-ui, sans-serif;
    font-size: .98rem; font-weight: 500; color: #9FC4B2;
    margin: 0 0 1.55rem; max-width: 56ch;
  }}

  .route {{ width: 100%; height: 66px; display: block; }}
  .route .track {{ stroke: rgba(255,255,255,.16); stroke-width: 2.5; }}
  .route .travelled {{
    stroke: #E8A33D; stroke-width: 3; stroke-linecap: round;
    stroke-dasharray: 620; stroke-dashoffset: 620;
    animation: travel 1.5s cubic-bezier(.33,.9,.35,1) .25s forwards;
  }}
  @keyframes travel {{ to {{ stroke-dashoffset: 0; }} }}
  .route .node {{ fill: #0F2A22; stroke-width: 2.5; }}
  .route .lbl {{
    font-family: Archivo, system-ui, sans-serif;
    font-size: 11px; font-weight: 600; fill: #7FAF99;
  }}
  .route .lbl-warn {{ fill: #D98A6A; }}

  .facts {{
    display: flex; gap: 2.6rem; flex-wrap: wrap;
    margin-top: 1.25rem; padding-top: 1.15rem;
    border-top: 1px solid rgba(255,255,255,.09);
  }}
  .fact {{ display: flex; flex-direction: column; gap: .12rem; }}
  .fact-n {{
    font-family: Archivo, system-ui, sans-serif;
    font-size: 1.32rem; font-weight: 700; color: #F2F7F4;
    font-variant-numeric: tabular-nums; line-height: 1;
  }}
  .fact-l {{
    font-family: Archivo, system-ui, sans-serif;
    font-size: .76rem; font-weight: 500; color: #7FAF99;
  }}

  @media (prefers-reduced-motion: reduce) {{
    .route .travelled {{ animation: none; stroke-dashoffset: 0; }}
    .hero-live i {{ animation: none; }}
  }}
  @media (max-width: 760px) {{
    .hero h1 {{ font-size: 1.85rem; }}
    .facts {{ gap: 1.4rem; }}
  }}
</style>

<div class="hero">
  <div class="hero-live"><i></i>Model live · scoring against held-out orders</div>
  <h1>E-commerce Delivery<br>Risk Prediction</h1>
  <p>Every order runs a clock. This surface reads how much of it is left, explains the reading, and prices what to do about it.</p>
  <svg class="route" viewBox="0 0 640 66" preserveAspectRatio="none" aria-hidden="true"><line class="track" x1="14" y1="26" x2="626" y2="26" /><line class="travelled" x1="14" y1="26" x2="470" y2="26" /><line x1="470" y1="14" x2="470" y2="38" stroke="#E8A33D" stroke-width="2" stroke-dasharray="3 3" /><circle class="node" cx="14" cy="26" r="5.5" stroke="#E8A33D" /><circle class="node" cx="470" cy="26" r="5.5" stroke="#E8A33D" /><circle class="node" cx="626" cy="26" r="5" stroke="#B3563A" /><text class="lbl" x="14" y="53">purchase</text><text class="lbl" x="470" y="53" text-anchor="middle">promised date</text><text class="lbl lbl-warn" x="626" y="53" text-anchor="end">late</text></svg>
  <div class="facts">{fact_html}</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("What this does, in plain terms"):
        st.markdown("""
This screen answers four questions about a single customer order, in the order a
decision actually needs them.

**1. Will it arrive late?** The model reads the order — where it\'s going, how far,
how quickly the seller handed it to the carrier, how much slack the delivery
promise leaves — and returns a probability. Around 8% of Olist orders arrive after
the date the customer was promised.

**2. Why does it think so?** Panel 1 lists the factors that pushed the estimate up
or down, written as sentences rather than feature names. Red raises risk, green
lowers it.

**3. What can we do about it?** Panels 5 and 8 let you change something you control
— dispatch speed, the length of the promise — and the model re-scores the order.
Panel 6 ranks the options.

**4. Is acting worth it?** Panel 9 puts a number on it: what a late delivery costs
on this order, what the intervention costs, and whether the difference is positive.

---

**Two things worth knowing before you act on a score.**

*Model confidence is not the same as risk.* Risk is how likely lateness is;
confidence is how much the estimate should be trusted. An order sitting right on
the action threshold gets low confidence even when the risk figure looks
unremarkable, because a small change would flip the recommendation. When
confidence is low the screen says so and explains why.

*The money is an assumption.* The cost figures in the sidebar are estimates, not
measured Olist finances. Every monetary number on the page moves when you move
them — which is why they\'re exposed rather than hidden in the code.
""")
    st.write("")

if view in ("Overview", "Explainability", "What-If Analysis"):
    pos = select_order()
    row, a = assess(pos)

    header_strip(pos, a)

    if a.probability >= service.threshold:
        st.warning(f"Above the {service.threshold:.0%} action threshold — likely to "
                   f"arrive after the promised date. Intervention recommended.")
    for w in a.warnings:
        st.caption(f"⚠️ {w}")
    if a.confidence.is_low:
        with st.expander("Why is this prediction uncertain?", expanded=True):
            for reason in a.confidence.reasons:
                st.markdown(f"- {reason}")

    st.divider()

    if view == "Overview":
        # Two panels per row rather than four. At four, the comparison table
        # was clipped and every label truncated; the density defeated the
        # point of the screen.
        for left_panel, right_panel in [
            (lambda: panel_why(a), lambda: panel_importance()),
            (lambda: panel_similar(a), lambda: panel_comparison(row, a)),
            (lambda: panel_whatif(row, a), lambda: panel_actions(a)),
            (lambda: panel_outcomes(a), lambda: panel_counterfactual(row, a)),
        ]:
            left, right = st.columns(2, gap="large")
            with left:
                with st.container(border=True):
                    left_panel()
            with right:
                with st.container(border=True):
                    right_panel()
            st.write("")

        with st.container(border=True):
            panel_impact(a)

    elif view == "Explainability":
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                panel_why(a)
            st.write("")
            with st.container(border=True):
                panel_importance()
        with right:
            with st.container(border=True):
                panel_similar(a)
            st.write("")
            with st.container(border=True):
                panel_comparison(row, a)
        st.write("")
        with st.expander("Audit trail — every statement and the attribution behind it"):
            st.dataframe(a.brief.audit(), hide_index=True, width="stretch")
            st.caption(f"Glossary coverage of shown drivers: "
                       f"{a.brief.glossary_coverage:.0%}")

    else:  # What-If Analysis
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                panel_whatif(row, a)
        with right:
            with st.container(border=True):
                panel_counterfactual(row, a)
        st.write("")
        with st.container(border=True):
            panel_impact(a)

    st.write("")
    if st.button("Export decision brief (PDF)", type="primary"):
        from olist.bll import build_report

        o = history.iloc[pos]
        context = {
            "Order ID": o["order_id"],
            "Route": f"{o.get('seller_state', '—')} → {o['customer_state']}",
            "Product category": o["product_category"],
            "Purchased": f"{o['order_purchase_timestamp']:%Y-%m-%d}",
            "Promised delivery": f"{o['order_estimated_delivery_date']:%Y-%m-%d}",
            "Order value": f"{currency} {float(o['total_price']):,.2f}",
        }
        with st.spinner("Building…"):
            path = Path(tempfile.gettempdir()) / f"brief_{str(o['order_id'])[:12]}.pdf"
            build_report(a, path, order_context=context, model_info=service.model_info())
        st.download_button("Download", path.read_bytes(), file_name=path.name,
                           mime="application/pdf")

elif view == "Risk Dashboard":
    st.markdown("## Risk Dashboard")
    flagged = risk_all >= service.threshold
    exposure = (history["total_price"] * (costs.refund_share + costs.churn_share)
                + costs.support_cost)

    # KPI cards with semantic colour coding:
    #   Blue  — neutral volume figure
    #   Amber — elevated risk count
    #   Red   — high-risk or loss figure
    #   Green — mean risk (informational, baseline context)
    kpi = [
        ("Orders scored",       f"{len(history):,}",                                  "#1A56DB", "#EBF5FF"),
        ("High risk (≥50%)",    f"{(risk_all >= 0.5).sum():,}",                       "#C77700", "#FDF3E3"),
        ("Flagged at threshold",f"{flagged.sum():,}  ({flagged.mean():.1%})",         "#B3261E", "#FBEAE8"),
        ("Mean risk",           f"{risk_all.mean():.1%}",                             "#1E7F4F", "#E8F5EE"),
        ("Expected loss, flagged", f"{currency} {(risk_all[flagged]*exposure[flagged]).sum():,.0f}", "#B3261E", "#FBEAE8"),
    ]
    k = st.columns(5)
    for col, (label, value, fg, bg) in zip(k, kpi):
        with col:
            col.markdown(
                f'''<div style="background:{bg};border-radius:10px;padding:1rem 1.1rem;
                    height:100%;border-left:4px solid {fg}">
                <div style="font-size:.78rem;color:#6B7A88;font-weight:600;
                    letter-spacing:.06em;text-transform:uppercase;margin-bottom:.35rem">
                {label}</div>
                <div style="font-size:1.55rem;font-weight:700;color:{fg};
                    font-variant-numeric:tabular-nums;line-height:1.1">
                {value}</div></div>''',
                unsafe_allow_html=True,
            )
    st.write("")

    st.divider()
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown(panel_head("A", "Risk distribution",
                               "How many orders sit in each risk band"),
                    unsafe_allow_html=True)
        # pd.cut yields Interval objects, which Altair cannot serialise, so
        # the bands become labels. They are zero-padded because the axis sorts
        # them as text: without padding "5%" lands after "45%".
        edges = np.arange(0, 1.05, 0.05)
        counts = (pd.cut(risk_all, edges).value_counts().sort_index()
                    .rename("orders"))
        counts.index = [f"{int(round(i.left * 100)):02d}%" for i in counts.index]
        counts.index.name = "risk band"
        st.bar_chart(counts, height=300)

        below = (risk_all < service.threshold).mean()
        top_band = counts.idxmax()
        p90 = np.percentile(risk_all, 90)
        st.markdown(
            f'<div class="caption"><b>How to read this.</b> Most orders cluster at '
            f'the low end — the tallest band is {top_band} and {below:.0%} of orders '
            f'fall below the {service.threshold:.0%} action threshold. The thin right '
            f'tail is where the operational work is: the riskiest 10% of orders start '
            f'at {p90:.0%}. A model that piled everything near the base rate would be '
            f'useless for triage; this one separates.</div>',
            unsafe_allow_html=True)
    with right, st.container(border=True):
        st.markdown(panel_head("B", "Mean risk over time"), unsafe_allow_html=True)
        m = (history.assign(risk=risk_all)
                    .set_index("order_purchase_timestamp")
                    .resample("ME")["risk"].agg(["mean", "size"]))
        series = m[m["size"] >= 30]["mean"]
        st.line_chart(series, height=300)

        if len(series) > 3:
            slope = np.polyfit(range(len(series)), series.to_numpy(), 1)[0]
            peak_month = series.idxmax()
            st.markdown(
                f'<div class="caption"><b>How to read this.</b> Predicted risk is not a '
                f'steady background rate — it arrives in shocks. Quiet months sit near '
                f'{series.min():.0%} while the worst, {peak_month:%B %Y}, reached '
                f'{series.max():.0%}, roughly {series.max() / max(series.min(), 0.01):.0f}x '
                f'higher. The trend across the period is '
                f'{"rising" if slope > 0 else "falling"} by {abs(slope) * 100:.2f} '
                f'percentage points a month. A fixed policy tuned for a quiet month '
                f'fails in a peak one, which is the case for scoring each order.</div>',
                unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown(panel_head("C", "Where risk concentrates"), unsafe_allow_html=True)
        tabs = st.tabs(["Customer state", "Product category", "Seller state"])
        for tab, col in zip(tabs, ["customer_state", "product_category", "seller_state"]):
            with tab:
                if col not in history.columns:
                    st.caption("Not available in this dataset.")
                    continue
                g = (history.assign(risk=risk_all).groupby(col)
                            .agg(mean_risk=("risk", "mean"), orders=("risk", "size"))
                            .query("orders >= 50")
                            .sort_values("mean_risk", ascending=False).head(12))
                if g.empty:
                    st.caption("Too few orders per group to summarise.")
                    continue
                st.bar_chart(g["mean_risk"], height=320)

                worst, best = g.index[0], g.index[-1]
                ratio = g["mean_risk"].iloc[0] / max(g["mean_risk"].iloc[-1], 0.001)
                label = col.replace("_", " ")
                st.markdown(
                    f'<div class="caption"><b>How to read this.</b> Risk is uneven across '
                    f'{label}. <b>{worst}</b> carries the highest mean risk at '
                    f'{g["mean_risk"].iloc[0]:.0%}, against {g["mean_risk"].iloc[-1]:.0%} '
                    f'for <b>{best}</b> — a {ratio:.1f}x spread. Groups shown have at '
                    f'least 50 orders, so these are not small-sample artefacts. This is '
                    f'where a fixed SLA is quietly failing one group while over-serving '
                    f'another.</div>',
                    unsafe_allow_html=True)

elif view == "Sales Analysis":
    st.markdown("## Sales Analysis")
    sales = history.copy()
    sales["revenue"] = sales["total_price"]
    sales["freight"] = sales["total_freight"]
    sales["month"] = sales["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    sales["quarter"] = sales["order_purchase_timestamp"].dt.quarter

    # --- quarter filter, as in the reference layout ---
    f1, f2 = st.columns([3, 1])
    quarters = f1.multiselect("Quarter", [1, 2, 3, 4], default=[1, 2, 3, 4],
                              format_func=lambda q: f"Qtr {q}",
                              label_visibility="collapsed")
    years = sorted(sales["order_purchase_timestamp"].dt.year.unique())
    year_pick = f2.selectbox("Year", ["All"] + [str(y) for y in years],
                             label_visibility="collapsed")

    view_df = sales[sales["quarter"].isin(quarters)] if quarters else sales
    if year_pick != "All":
        view_df = view_df[view_df["order_purchase_timestamp"].dt.year == int(year_pick)]

    if view_df.empty:
        st.warning("No orders match this filter.")
        st.stop()

    st.write("")

    # --- KPI row ---
    k = st.columns(5)
    k[0].metric("Revenue", f"{currency} {view_df['revenue'].sum():,.0f}")
    k[1].metric("Items sold", f"{int(view_df['n_items'].sum()):,}")
    k[2].metric("Orders", f"{len(view_df):,}")
    k[3].metric("Average order value", f"{currency} {view_df['revenue'].mean():,.0f}")
    k[4].metric("Freight collected", f"{currency} {view_df['freight'].sum():,.0f}")

    st.markdown(
        '<div class="caption"><b>On profit.</b> The Olist release records what the '
        'customer paid — item price and freight — but no cost of goods or carrier '
        'cost. Profit therefore cannot be computed, so the panels below report '
        '<b>revenue</b>. Substituting an assumed margin would produce a plausible '
        'number with nothing behind it.</div>', unsafe_allow_html=True)

    st.divider()

    # --- Row 1: revenue over time (full width, the anchor chart) ---
    with st.container(border=True):
        st.markdown(panel_head("S1", "Revenue by month",
                               "Where the money actually came from"),
                    unsafe_allow_html=True)
        monthly = view_df.groupby("month").agg(
            revenue=("revenue", "sum"), orders=("revenue", "size")).reset_index()
        st.bar_chart(monthly.set_index("month")["revenue"], height=300)

        if monthly.empty or monthly["revenue"].sum() == 0:
            st.info("No revenue recorded in this window.")
        else:
            peak = monthly.loc[monthly["revenue"].idxmax()]
            share = peak["revenue"] / monthly["revenue"].sum()
            st.markdown(
                f'<div class="caption"><b>How to read this.</b> The strongest month '
                f'was <b>{peak["month"]:%B %Y}</b> at '
                f'{currency} {peak["revenue"]:,.0f}, {share:.0%} of the period\'s '
                f'revenue on its own. Volume grew steeply across the window, so a '
                f'flat monthly comparison understates how much later months carry.'
                f'</div>', unsafe_allow_html=True)

    st.write("")

    # --- Row 2: states and payment mix ---
    left, right = st.columns(2, gap="large")

    with left, st.container(border=True):
        st.markdown(panel_head("S2", "Top states by revenue",
                               "Where the customers are"), unsafe_allow_html=True)
        by_state = (view_df.groupby("customer_state")["revenue"].sum()
                           .sort_values(ascending=False).head(10))
        if by_state.empty or view_df["revenue"].sum() == 0:
            st.info("No revenue recorded in this window.")
        else:
            st.bar_chart(by_state, horizontal=True, height=320)
            top_share = by_state.iloc[0] / view_df["revenue"].sum()
            st.markdown(
                f'<div class="caption"><b>How to read this.</b> '
                f'<b>{by_state.index[0]}</b> alone accounts for {top_share:.0%} of '
                f'revenue, and the top three for '
                f'{by_state.head(3).sum() / view_df["revenue"].sum():.0%}. That '
                f'concentration is a logistics fact as much as a commercial one: most '
                f'parcels start and end in the same few states.</div>',
                unsafe_allow_html=True)

    with right, st.container(border=True):
        st.markdown(panel_head("S3", "Orders by payment method",
                               "How customers pay"), unsafe_allow_html=True)
        pay = (view_df["payment_type"].dropna().value_counts(normalize=True)
                      .rename_axis("method").reset_index(name="share"))
        if pay.empty:
            st.info("No payment methods recorded in this window.")
            pay = None
        else:
            donut = (alt.Chart(pay)
                    .mark_arc(innerRadius=62, outerRadius=105)
                    .encode(theta=alt.Theta("share:Q", stack=True),
                            color=alt.Color("method:N", title=None,
                                            scale=alt.Scale(scheme="tealblues")),
                            tooltip=["method:N",
                                     alt.Tooltip("share:Q", format=".1%")])
                        .properties(height=300))
            st.altair_chart(donut, use_container_width=True)

            lead = pay.iloc[0]
            inst = view_df.loc[view_df["payment_type"] == "credit_card",
                               "max_installments"]
            multi = (inst > 1).mean() if len(inst) else 0
            st.markdown(
                f'<div class="caption"><b>How to read this.</b> '
                f'<b>{lead["method"]}</b> carries {lead["share"]:.0%} of orders. Of '
                f'those paid by card, {multi:.0%} were split across instalments — so '
                f'cash actually reaches the platform over months, not at the order '
                f'date. That lag belongs in working-capital planning, not just the '
                f'revenue line.</div>', unsafe_allow_html=True)

    st.write("")

    # --- Row 3: categories and customers ---
    left, right = st.columns(2, gap="large")

    with left, st.container(border=True):
        st.markdown(panel_head("S4", "Revenue by category",
                               "What sells"), unsafe_allow_html=True)
        by_cat = (view_df.groupby("product_category")["revenue"].sum()
                         .sort_values(ascending=False).head(8)
                         .rename_axis("category").reset_index(name="revenue"))
        if by_cat.empty or by_cat["revenue"].sum() == 0:
            st.info("No category revenue recorded in this window.")
        else:
            cat_donut = (alt.Chart(by_cat)
                        .mark_arc(innerRadius=62, outerRadius=105)
                        .encode(theta=alt.Theta("revenue:Q", stack=True),
                                color=alt.Color("category:N", title=None,
                                                scale=alt.Scale(scheme="tealblues")),
                                tooltip=["category:N",
                                         alt.Tooltip("revenue:Q", format=",.0f")])
                            .properties(height=300))
            st.altair_chart(cat_donut, use_container_width=True)
            st.markdown(
                f'<div class="caption"><b>How to read this.</b> The top eight '
                f'categories shown carry '
                f'{by_cat["revenue"].sum() / view_df["revenue"].sum():.0%} of revenue, '
                f'led by <b>{by_cat.iloc[0]["category"]}</b>. Worth cross-reading '
                f'against the Risk Dashboard: a high-revenue category with high '
                f'delivery risk is where a fix pays back fastest.</div>',
                unsafe_allow_html=True)

    with right, st.container(border=True):
        st.markdown(panel_head("S5", "Top customers by spend",
                               "Olist data is anonymised, so these are IDs"),
                    unsafe_allow_html=True)
        by_cust = None
        if "customer_unique_id" in view_df.columns:
            by_cust = (view_df.groupby("customer_unique_id")
                              .agg(spend=("revenue", "sum"), orders=("revenue", "size"))
                              .sort_values("spend", ascending=False).head(8))

        if by_cust is None:
            st.info("Customer identifiers are not present in this table.")
        elif by_cust.empty:
            st.info("No customers recorded in this window.")
        else:
            chart = by_cust["spend"].copy()
            chart.index = [c[:10] + "…" for c in chart.index]
            st.bar_chart(chart, horizontal=True, height=320)

            repeat = (view_df.groupby("customer_unique_id").size() > 1).mean()
            st.markdown(
                f'<div class="caption"><b>How to read this.</b> Only <b>{repeat:.1%}</b> '
                f'of customers in this window ordered more than once, so revenue rests '
                f'on acquisition rather than repeat business. That is what makes late '
                f'delivery expensive: a first order that arrives late rarely gets a '
                f'second chance to recover the relationship.</div>',
                unsafe_allow_html=True)

    st.write("")

    # --- Row 4: the bridge back to the risk work ---
    with st.container(border=True):
        st.markdown(panel_head("S6", "Revenue exposed to delivery risk",
                               "Where the sales view meets the risk model"),
                    unsafe_allow_html=True)
        exposed = view_df.assign(risk=risk_all[view_df.index])
        # The minimum-orders floor scales with the filtered window. A fixed
        # floor of 100 empties the table entirely for a short period such as
        # 2016, which holds only a few hundred orders in total.
        min_orders = max(10, min(100, len(view_df) // 40))
        by_cat_risk = (exposed.groupby("product_category")
                              .agg(revenue=("revenue", "sum"),
                                   mean_risk=("risk", "mean"),
                                   orders=("revenue", "size"))
                              .query("orders >= @min_orders")
                              .assign(at_risk=lambda d: d["revenue"] * d["mean_risk"])
                              .sort_values("at_risk", ascending=False).head(10))

        if by_cat_risk.empty:
            st.info(
                f"No product category in this window has at least {min_orders} orders, "
                f"so a per-category exposure figure would rest on too few observations "
                f"to be meaningful. Widen the filter to see this panel."
            )
        else:
            st.dataframe(
                by_cat_risk.assign(
                    revenue=lambda d: d["revenue"].round(0),
                    mean_risk=lambda d: (d["mean_risk"] * 100).round(1),
                    at_risk=lambda d: d["at_risk"].round(0),
                ).rename(columns={"revenue": f"Revenue ({currency})",
                                  "mean_risk": "Mean risk (%)",
                                  "orders": "Orders",
                                  "at_risk": f"Revenue at risk ({currency})"}),
                width="stretch", height=320,
            )
            st.markdown(
                f'<div class="caption"><b>How to read this.</b> Revenue multiplied by '
                f'mean predicted late-delivery risk, per category. This is the ranking '
                f'that matters operationally: not the riskiest categories, and not the '
                f'largest, but the ones where the two overlap. '
                f'<b>{by_cat_risk.index[0]}</b> tops it at '
                f'{currency} {by_cat_risk["at_risk"].iloc[0]:,.0f} exposed. '
                f'Categories with fewer than {min_orders} orders in this window are '
                f'excluded.</div>', unsafe_allow_html=True)

elif view == "Orders":
    st.markdown("## Orders")
    st.write("")
    with st.container(border=True):
        st.markdown(panel_head("—", "Order queue",
                           "Sorted by expected loss: what to call first"),
                unsafe_allow_html=True)
    exposure = (history["total_price"] * (costs.refund_share + costs.churn_share)
                + costs.support_cost)
    table = (history.assign(risk=risk_all, expected_loss=risk_all * exposure)
                    .sort_values("expected_loss", ascending=False).head(60))
    st.dataframe(
        pd.DataFrame({
            "Order": table["order_id"].str[:16],
            "State": table["customer_state"],
            "Category": table["product_category"],
            "Purchased": table["order_purchase_timestamp"].dt.date,
            "Risk": table["risk"],
            "Level": [service.risk_level(r) for r in table["risk"]],
            f"Value ({currency})": table["total_price"].round(2),
            f"Expected loss ({currency})": table["expected_loss"].round(2),
            "Action": np.where(table["risk"] >= service.threshold, "Escalate", "Standard"),
        }),
        hide_index=True, width="stretch", height=560,
        column_config={"Risk": st.column_config.ProgressColumn(
            "Risk", format="%.0f%%", min_value=0.0, max_value=1.0)},
    )

else:  # Model Insights
    st.markdown("## Model Insights")
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown(panel_head("—", "Model card",
                               "Only metrics recorded in the trained artefact"),
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(service.model_info().items(),
                                  columns=["Field", "Value"]),
                     hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.markdown(panel_head("—", "What this dataset supports",
                               "Capabilities are disclosed, not silently omitted"),
                    unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{"Capability": n, "Available": "yes" if c.available else "no",
                           "Reason / requirement": c.reason or "—"}
                          for n, c in service.available_levers().items()]),
            hide_index=True, width="stretch",
        )
    st.write("")
    panel_importance()

st.markdown('<div class="footer">All predictions are generated by a model trained on '
            'historical data and are decision support, not certainty. Monetary figures '
            'rest on configured cost assumptions, not measured finances.</div>',
            unsafe_allow_html=True)
