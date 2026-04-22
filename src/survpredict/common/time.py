"""Time helpers. Keep everything UTC and tz-aware."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def floor_to(ts: datetime, seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=timezone.utc)


def window_bounds(end: datetime, seconds: int) -> tuple[datetime, datetime]:
    return end - timedelta(seconds=seconds), end
