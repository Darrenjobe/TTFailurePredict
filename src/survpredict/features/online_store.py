"""Online feature store (Redis).

Redis keys:
  - ``f:{guid}:{feature}:{window}``   -> ZSET of (value, timestamp)
  - ``fvec:{guid}``                    -> HASH of latest derived feature values

The ZSETs are trimmed to N=60 samples per (guid, feature, window). The HASH is
what the inference service reads on the hot path.
"""

from __future__ import annotations

import time
from typing import Iterable

from survpredict.common.db import redis_client

MAX_SAMPLES = 60


def push_sample(
    entity_guid: str, feature_name: str, window_seconds: int, value: float, ts: float
) -> None:
    r = redis_client()
    key = f"f:{entity_guid}:{feature_name}:{window_seconds}"
    pipe = r.pipeline()
    pipe.zadd(key, {f"{value}:{ts}": ts})
    pipe.zremrangebyrank(key, 0, -MAX_SAMPLES - 1)
    pipe.expire(key, 3600 * 6)
    pipe.execute()


def write_feature_vector(entity_guid: str, values: dict[str, float]) -> None:
    r = redis_client()
    key = f"fvec:{entity_guid}"
    pipe = r.pipeline()
    pipe.hset(key, mapping={k: str(v) for k, v in values.items()})
    pipe.hset(key, "_updated_at", str(time.time()))
    pipe.expire(key, 3600 * 2)
    pipe.execute()


def read_feature_vector(entity_guid: str) -> dict[str, float]:
    r = redis_client()
    raw = r.hgetall(f"fvec:{entity_guid}")
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def bump_dirty(entity_guid: str) -> None:
    """Mark an entity as needing re-scoring on the next warm sweep."""
    redis_client().sadd("dirty_entities", entity_guid)


def drain_dirty() -> list[str]:
    r = redis_client()
    items = r.smembers("dirty_entities")
    if items:
        r.delete("dirty_entities")
    return list(items)


def mark_active(entity_guid: str) -> None:
    redis_client().sadd("active_entities", entity_guid)


def active_entities() -> list[str]:
    return list(redis_client().smembers("active_entities"))


def bulk_set_feature_vectors(updates: Iterable[tuple[str, dict[str, float]]]) -> int:
    r = redis_client()
    pipe = r.pipeline()
    n = 0
    for guid, values in updates:
        key = f"fvec:{guid}"
        pipe.hset(key, mapping={k: str(v) for k, v in values.items()})
        pipe.expire(key, 3600 * 2)
        n += 1
    pipe.execute()
    return n
