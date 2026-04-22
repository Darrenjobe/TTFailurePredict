"""LLM-based postmortem structurer.

Takes raw postmortem text and asks Claude to extract a controlled-vocabulary
JSON structure (design doc §9.1 and §15). Pinning the model is a stability
requirement -- labels produced by a newer model version must not silently
drift the training set. See ``ANTHROPIC_MODEL`` in the config.

The LLM never writes to the training labels directly -- it produces the
structured form, which is then inserted into `postmortems` and drives the
label reconciliation job.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.config import settings

log = get_logger(__name__)

ROOT_CAUSE_VOCAB = [
    "deploy_regression",
    "resource_saturation",
    "dependency_failure",
    "config_change",
    "traffic_spike",
    "third_party_outage",
    "data_issue",
    "infra_failure",
    "security_incident",
    "unknown",
]

STRUCTURING_PROMPT = """You structure engineering postmortems into JSON.

Output ONLY a JSON object (no prose, no markdown) matching this schema:
{{
  "summary": str,
  "event_start": ISO-8601 or null,
  "event_end": ISO-8601 or null,
  "entities_affected": [str],       // best-effort NR entity names or guids
  "root_cause_category": one of {vocab},
  "severity": "sev1" | "sev2" | "sev3" | "sev4" | "unknown",
  "contributing_signals": [str],    // feature-like names or metric names
  "human_impact": str,
  "detection_source": str,
  "timeline": [{{"at": ISO-8601, "what": str}}]
}}

Unknown values use null for scalars and [] for arrays. Do NOT invent data.

POSTMORTEM:
---
{text}
---
"""


def _client() -> Anthropic:
    s = settings()
    if not s.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return Anthropic(api_key=s.anthropic_api_key)


def structure(raw_text: str) -> dict[str, Any]:
    s = settings()
    client = _client()
    prompt = STRUCTURING_PROMPT.format(vocab=ROOT_CAUSE_VOCAB, text=raw_text)

    resp = client.messages.create(
        model=s.anthropic_model,
        max_tokens=2048,
        temperature=0,
        system="You are a careful postmortem structurer. Output valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("structurer_invalid_json", err=str(e), preview=text[:200])
        raise
    return parsed


def structure_and_persist(postmortem_id: str) -> dict[str, Any]:
    with pg_cursor() as cur:
        cur.execute("SELECT raw_text FROM postmortems WHERE postmortem_id = %s", (postmortem_id,))
        row = cur.fetchone()
    if not row:
        raise ValueError(f"postmortem {postmortem_id} not found")

    structured = structure(row["raw_text"])
    with pg_cursor() as cur:
        cur.execute(
            """
            UPDATE postmortems
            SET structured = %s,
                entities_affected = %s,
                root_cause_category = %s,
                severity = %s,
                event_start = %s,
                event_end = %s,
                contributing_signals = %s
            WHERE postmortem_id = %s
            """,
            (
                json.dumps(structured),
                structured.get("entities_affected") or [],
                structured.get("root_cause_category"),
                structured.get("severity"),
                structured.get("event_start"),
                structured.get("event_end"),
                structured.get("contributing_signals") or [],
                postmortem_id,
            ),
        )
    log.info("postmortem_structured", id=postmortem_id)
    return structured


def structure_pending(limit: int = 10) -> int:
    """Process any postmortems that don't yet have a structured form."""
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT postmortem_id FROM postmortems
            WHERE structured IS NULL
            ORDER BY created_at LIMIT %s
            """,
            (limit,),
        )
        ids = [r["postmortem_id"] for r in cur.fetchall()]
    for pid in ids:
        try:
            structure_and_persist(str(pid))
        except Exception as e:
            log.warning("structure_failed", id=str(pid), err=str(e))
    return len(ids)
