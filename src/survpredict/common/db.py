"""Postgres + Redis connection helpers.

Thin layer over psycopg (sync) and redis-py. Intentionally avoids SQLAlchemy
ORM in the hot path -- we use raw SQL everywhere that perf matters and keep
the API small.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
import redis

from survpredict.config import settings

_pg_pool: ConnectionPool | None = None
_redis_client: redis.Redis | None = None


def pg_pool() -> ConnectionPool:
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = ConnectionPool(
            conninfo=settings().pg_dsn,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pg_pool


@contextmanager
def pg_conn() -> Iterator[psycopg.Connection]:
    with pg_pool().connection() as conn:
        yield conn


@contextmanager
def pg_cursor() -> Iterator[psycopg.Cursor]:
    with pg_conn() as conn, conn.cursor() as cur:
        yield cur


def redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(settings().redis_url, decode_responses=True)
    return _redis_client
