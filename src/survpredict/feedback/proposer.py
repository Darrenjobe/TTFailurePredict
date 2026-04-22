"""LLM-assisted feature proposal queue (design doc §9.3 and §9.5).

For each false-negative event we assemble:
  - the postmortem narrative
  - the full set of feature spec names available at prediction time
  - the recent telemetry for the affected entities

and ask Claude to propose new features (as YAML), new edge types, or new event
types. Proposals land in the ``feature_proposals`` table with status=pending;
a human must approve before anything touches the production feature specs.
"""

from __future__ import annotations

import json
from datetime import timedelta

from anthropic import Anthropic

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.config import settings
from survpredict.features.offline_store import load_feature_matrix
from survpredict.features.spec import load_feature_specs

log = get_logger(__name__)

PROPOSER_SYSTEM = """You propose new observability features for a survival model.
You do not implement. You do not edit anything. You output proposals only.
Output must be valid YAML matching the existing feature_specs/ format.
"""

PROPOSER_PROMPT = """A survival model for predictive observability missed the
following event. Your job is to propose *new* features, entity-edge types, or
event types that might have caught it.

CURRENT FEATURE NAMES (do not duplicate):
{feature_names}

POSTMORTEM (structured):
{postmortem_json}

RECENT TELEMETRY (long-form, sampled):
{telemetry_sample}

Output YAML with one or more entries in this shape:
- name: <snake_case_name>
  description: <one sentence>
  entity_classes: [<one or more>]
  source: nr_metric | nr_event | derived
  nrql: "<NRQL with ? placeholder for entity.guid>"   # if source is nr_metric / nr_event
  windows: [<seconds>]
  transformations: [level, trend_slope, volatility_stddev, changepoint_distance]
  rationale: <short rationale tying back to the postmortem>

Do NOT output anything other than the YAML block.
"""


def propose_for_event(event_id: str) -> str | None:
    s = settings()
    if not s.anthropic_api_key:
        log.warning("proposer_disabled_no_key")
        return None

    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT e.entity_guid, e.entity_class, e.occurred_at, p.structured
            FROM events e
            LEFT JOIN postmortems p ON p.postmortem_id = e.postmortem_id
            WHERE e.event_id = %s
            """,
            (event_id,),
        )
        row = cur.fetchone()
    if not row:
        return None

    specs = load_feature_specs()
    feature_names = [s.name for s in specs]
    since = row["occurred_at"] - timedelta(minutes=30)
    telemetry = load_feature_matrix(
        row["entity_class"], feature_names, since=since, until=row["occurred_at"]
    )
    telemetry_sample = telemetry.head(50).to_csv(index=False) if not telemetry.empty else "(no telemetry)"

    prompt = PROPOSER_PROMPT.format(
        feature_names=json.dumps(feature_names),
        postmortem_json=json.dumps(row.get("structured") or {}, indent=2),
        telemetry_sample=telemetry_sample,
    )

    client = Anthropic(api_key=s.anthropic_api_key)
    resp = client.messages.create(
        model=s.anthropic_model,
        max_tokens=2048,
        temperature=0,
        system=PROPOSER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    yaml_text = "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    )

    with pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO feature_proposals (yaml_spec, rationale, status, triggered_by)
            VALUES (%s, %s, 'pending',
              (SELECT postmortem_id FROM events WHERE event_id = %s))
            RETURNING proposal_id
            """,
            (yaml_text, f"auto-proposed for missed event {event_id}", event_id),
        )
        pid = cur.fetchone()["proposal_id"]
    log.info("proposal_created", proposal_id=str(pid), event_id=event_id)
    return str(pid)
