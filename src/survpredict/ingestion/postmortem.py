"""Postmortem ingester.

Supports three sources in v1:
  - markdown directory (watch for new/changed files)
  - slack channel (noop stub -- plug in your slack client)
  - jira project (noop stub)

Each new postmortem is inserted raw; the feedback layer (``survpredict.feedback``)
picks it up and runs LLM structuring.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class RawPostmortem:
    source: str
    source_ref: str
    raw_text: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            f"{self.source}:{self.source_ref}:{self.raw_text[:512]}".encode()
        ).hexdigest()


def ingest_markdown_dir(root: Path) -> int:
    """Scan a directory for .md files; insert any that aren't already stored."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    new = 0
    for md in sorted(root.rglob("*.md")):
        pm = RawPostmortem(
            source="markdown",
            source_ref=str(md.relative_to(root)),
            raw_text=md.read_text(encoding="utf-8"),
        )
        if _already_stored(pm):
            continue
        _insert(pm)
        new += 1
    log.info("postmortems_ingested", source="markdown", count=new, root=str(root))
    return new


def _already_stored(pm: RawPostmortem) -> bool:
    with pg_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM postmortems WHERE source = %s AND source_ref = %s LIMIT 1",
            (pm.source, pm.source_ref),
        )
        return cur.fetchone() is not None


def _insert(pm: RawPostmortem) -> str:
    with pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO postmortems (source, source_ref, raw_text)
            VALUES (%s, %s, %s)
            RETURNING postmortem_id
            """,
            (pm.source, pm.source_ref, pm.raw_text),
        )
        row = cur.fetchone()
        return str(row["postmortem_id"])


def ingest_slack(_channel: str) -> int:
    """Slack ingestion stub. Wire in your slack client here."""
    log.warning("slack_ingester_stub", msg="Slack ingester not implemented in v1")
    return 0


def ingest_jira(_project: str) -> int:
    """Jira ingestion stub."""
    log.warning("jira_ingester_stub", msg="Jira ingester not implemented in v1")
    return 0
