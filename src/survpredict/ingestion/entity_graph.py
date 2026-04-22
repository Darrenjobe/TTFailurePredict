"""Entity relationship snapshotter.

Pulls the full entity relationship graph from NerdGraph every N minutes and
stores it as an adjacency list with temporal validity (design doc §5.3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.common.time import utcnow
from survpredict.ingestion.newrelic_client import NewRelicClient

log = get_logger(__name__)

RELATIONSHIPS_QUERY = """
query($guids: [EntityGuid]!) {
  actor {
    entities(guids: $guids) {
      guid
      name
      entityType
      relatedEntities {
        results {
          type
          source { entity { guid name entityType } }
          target { entity { guid name entityType } }
        }
      }
    }
  }
}
"""

ENTITY_SEARCH_QUERY = """
query($query: String!, $cursor: String) {
  actor {
    entitySearch(query: $query) {
      count
      results(cursor: $cursor) {
        nextCursor
        entities { guid name entityType tags { key values } }
      }
    }
  }
}
"""


def search_entities(entity_type: str) -> list[dict[str, Any]]:
    """Page through entitySearch for a given type (e.g. 'APPLICATION')."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    with NewRelicClient() as client:
        while True:
            data = client.graphql(
                ENTITY_SEARCH_QUERY,
                {"query": f"type = '{entity_type}'", "cursor": cursor},
            )
            results = data["actor"]["entitySearch"]["results"]
            out.extend(results["entities"])
            cursor = results["nextCursor"]
            if not cursor:
                break
    return out


def upsert_entities(entities: list[dict[str, Any]], entity_class: str) -> int:
    now = utcnow()
    with pg_cursor() as cur:
        for e in entities:
            cur.execute(
                """
                INSERT INTO entities (entity_guid, entity_class, name, tags, last_seen_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (entity_guid) DO UPDATE
                SET name = EXCLUDED.name,
                    tags = EXCLUDED.tags,
                    last_seen_at = EXCLUDED.last_seen_at
                """,
                (e["guid"], entity_class, e.get("name"), _tags_to_jsonb(e.get("tags", [])), now),
            )
    return len(entities)


def _tags_to_jsonb(tags: list[dict[str, Any]]) -> str:
    import json

    return json.dumps({t["key"]: t["values"] for t in tags or []})


def snapshot_relationships(guids: list[str]) -> int:
    """Pull relationships for a batch of entity guids and upsert the adjacency list."""
    now = utcnow()
    edges_written = 0
    if not guids:
        return 0
    with NewRelicClient() as client:
        for batch in _chunks(guids, 25):  # NerdGraph caps at 25 per call
            try:
                data = client.graphql(RELATIONSHIPS_QUERY, {"guids": batch})
            except Exception as e:
                log.warning("relationships_batch_failed", err=str(e), size=len(batch))
                continue
            for entity in data["actor"]["entities"] or []:
                related = (entity.get("relatedEntities") or {}).get("results") or []
                for rel in related:
                    src = (rel.get("source") or {}).get("entity") or {}
                    dst = (rel.get("target") or {}).get("entity") or {}
                    if not src.get("guid") or not dst.get("guid"):
                        continue
                    _upsert_edge(src["guid"], dst["guid"], rel.get("type") or "related", now)
                    edges_written += 1
    log.info("edges_snapshot_complete", edges=edges_written)
    return edges_written


def _upsert_edge(src: str, dst: str, relationship: str, valid_from: datetime) -> None:
    with pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_edges (src_guid, dst_guid, relationship, weight, valid_from)
            VALUES (%s, %s, %s, 1.0, %s)
            ON CONFLICT (src_guid, dst_guid, relationship, valid_from) DO NOTHING
            """,
            (src, dst, relationship, valid_from),
        )


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
