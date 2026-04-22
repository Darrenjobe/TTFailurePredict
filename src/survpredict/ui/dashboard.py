"""Streamlit risk dashboard.

Run with: ``streamlit run src/survpredict/ui/dashboard.py``.

Shows:
  - Top-N riskiest entities right now
  - Hazard over time for a selected entity
  - Top contributing features
  - Downstream dependency risk propagation
  - Recent false negatives awaiting feature proposals
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from survpredict.common.db import pg_cursor
from survpredict.config import settings


st.set_page_config(page_title="survpredict", layout="wide")
st.title("Predictive Observability — Risk Dashboard")

s = settings()

with st.sidebar:
    window_min = st.slider("Lookback (minutes)", 5, 240, 60)
    top_k = st.slider("Top-K entities", 5, 100, 25)
    hazard_threshold = st.slider(
        "Hazard threshold", 0.0, 1.0, float(s.propagation_hazard_threshold), 0.05
    )


@st.cache_data(ttl=15)
def load_top(k: int, minutes: int) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (p.entity_guid)
                p.entity_guid, e.name, e.entity_class, p.predicted_at,
                p.hazard_score, p.survival_5min, p.survival_15min,
                p.survival_30min, p.survival_60min, p.top_features,
                p.dep_risk_delta, p.model_version
            FROM predictions p
            LEFT JOIN entities e ON e.entity_guid = p.entity_guid
            WHERE p.predicted_at >= %s
            ORDER BY p.entity_guid, p.predicted_at DESC
            """,
            (cutoff,),
        )
        df = pd.DataFrame(cur.fetchall())
    if df.empty:
        return df
    df = df.sort_values("hazard_score", ascending=False).head(k)
    return df


@st.cache_data(ttl=15)
def load_history(guid: str, minutes: int) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(minutes=minutes * 10)
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT predicted_at, hazard_score, survival_15min, survival_60min,
                   dep_risk_delta
            FROM predictions
            WHERE entity_guid = %s AND predicted_at >= %s
            ORDER BY predicted_at
            """,
            (guid, cutoff),
        )
        return pd.DataFrame(cur.fetchall())


@st.cache_data(ttl=30)
def load_false_negatives() -> pd.DataFrame:
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT event_id, entity_guid, entity_class, occurred_at, event_type
            FROM events
            WHERE metadata ? 'miss' AND (metadata->>'miss')::bool = TRUE
            ORDER BY occurred_at DESC
            LIMIT 50
            """
        )
        return pd.DataFrame(cur.fetchall())


top = load_top(top_k, window_min)

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Top-K riskiest entities")
    if top.empty:
        st.info("No predictions in the selected window yet.")
    else:
        display = top[[
            "entity_guid", "name", "entity_class", "hazard_score",
            "survival_15min", "survival_60min", "dep_risk_delta",
            "predicted_at", "model_version",
        ]].copy()
        display["hazard_score"] = display["hazard_score"].round(3)
        display[["survival_15min", "survival_60min"]] = display[
            ["survival_15min", "survival_60min"]
        ].round(3)
        st.dataframe(display, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Hazard distribution")
    if not top.empty:
        fig = px.histogram(top, x="hazard_score", nbins=20)
        fig.add_vline(x=hazard_threshold, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Per-entity drill-down")

if not top.empty:
    choice = st.selectbox(
        "Entity",
        options=top["entity_guid"].tolist(),
        format_func=lambda g: f"{g[:12]}... -- {top.loc[top.entity_guid==g, 'name'].iloc[0]}",
    )
    hist = load_history(choice, window_min)
    if not hist.empty:
        fig = px.line(
            hist,
            x="predicted_at",
            y=["hazard_score", "survival_15min", "survival_60min"],
            title=f"Hazard / survival over time for {choice[:12]}...",
        )
        st.plotly_chart(fig, use_container_width=True)

    latest = top[top.entity_guid == choice].iloc[0]
    feats = latest.get("top_features")
    if feats:
        parsed = json.loads(feats) if isinstance(feats, str) else feats
        fdf = pd.DataFrame(parsed, columns=["feature", "contribution"])
        fdf = fdf.sort_values("contribution", key=abs, ascending=False).head(10)
        st.plotly_chart(
            px.bar(fdf, x="contribution", y="feature", orientation="h",
                   title="Top contributing features (SHAP)"),
            use_container_width=True,
        )

st.divider()
st.subheader("False negatives awaiting feature proposals")
fn = load_false_negatives()
if fn.empty:
    st.success("No unprocessed misses.")
else:
    st.dataframe(fn, use_container_width=True, hide_index=True)
