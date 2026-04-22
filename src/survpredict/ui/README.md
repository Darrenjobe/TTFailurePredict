# `survpredict.ui`

v1 presentation layer. Streamlit for fast iteration; React is a v2
concern.

> Design doc: [§4.2 Presentation + feedback](../../../docs/DESIGN.md#presentation--feedback).

## Run

```bash
streamlit run src/survpredict/ui/dashboard.py
# or via docker compose:
docker compose up -d dashboard
open http://localhost:8501
```

## Panels

1. **Top-K riskiest entities** — the operational list. Sorted by hazard
   over a user-configurable lookback window.
2. **Hazard distribution** — histogram across the current top list with
   the configured propagation threshold overlaid.
3. **Per-entity drill-down** — time series of hazard / survival +
   top-SHAP attributions for the selected entity.
4. **False negatives awaiting feature proposals** — operational queue
   fed by `feedback.reconcile`.

All queries are read-only against Postgres. Streamlit's `@st.cache_data`
caches them for 15–30 seconds so refreshes don't hammer the DB.

## Why Streamlit in v1?

It's the fastest way to put real data in front of real humans. Every
decision about what to visualize is still churning — Streamlit lets us
change layout faster than we can change a React bundle. Once the panels
and interactions settle, porting to a proper frontend is a mechanical
exercise.

## Extending

- **New panel:** add a cached loader and a chart block. Prefer Plotly
  for interactivity.
- **Auth:** none in v1. When we stand this up outside a private network,
  wrap behind an OAuth proxy rather than baking auth into the app.
- **Alerting from the dashboard:** don't. Use the NR publish path —
  that's what `PredictedRisk` is for.
