"""Graph propagation.

When an entity's hazard crosses the propagation threshold, walk up to N hops
downstream in the dependency graph and update their upstream-hazard feature.
A rate limiter prevents propagation storms.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from survpredict.common.db import pg_cursor, redis_client
from survpredict.common.logging import get_logger
from survpredict.config import settings

log = get_logger(__name__)

PROPAGATION_LIMITER_KEY = "propagation:rate"
PROPAGATION_LIMIT_PER_MIN = 500


def propagate(root_guid: str, root_hazard: float) -> list[str]:
    """Return the list of entity guids whose dep_risk was updated."""
    s = settings()
    if root_hazard < s.propagation_hazard_threshold:
        return []
    if not _claim_budget():
        log.warning("propagation_rate_limited", guid=root_guid)
        return []

    affected: list[str] = []
    visited: set[str] = {root_guid}
    queue: deque[tuple[str, int, float]] = deque([(root_guid, 0, root_hazard)])

    while queue:
        guid, hop, hazard = queue.popleft()
        if hop >= s.propagation_max_hops:
            continue
        for dst, weight in _downstream(guid):
            if dst in visited:
                continue
            visited.add(dst)
            contribution = weight * hazard * (0.6 ** hop)
            _bump_dep_risk(dst, contribution)
            affected.append(dst)
            queue.append((dst, hop + 1, hazard * 0.6))

    log.info("propagation", root=root_guid, affected=len(affected))
    return affected


def _downstream(guid: str) -> Iterable[tuple[str, float]]:
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT dst_guid, weight
            FROM entity_edges
            WHERE src_guid = %s AND valid_to IS NULL
            """,
            (guid,),
        )
        for r in cur.fetchall():
            yield r["dst_guid"], float(r.get("weight") or 1.0)


def _bump_dep_risk(guid: str, delta: float) -> None:
    r = redis_client()
    r.incrbyfloat(f"deprisk:{guid}", delta)
    r.expire(f"deprisk:{guid}", 600)
    r.sadd("dirty_entities", guid)


def _claim_budget() -> bool:
    r = redis_client()
    pipe = r.pipeline()
    pipe.incr(PROPAGATION_LIMITER_KEY)
    pipe.expire(PROPAGATION_LIMITER_KEY, 60)
    count, _ = pipe.execute()
    return int(count) <= PROPAGATION_LIMIT_PER_MIN
