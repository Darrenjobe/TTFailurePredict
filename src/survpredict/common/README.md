# `survpredict.common`

Cross-cutting utilities: settings, DB access, logging, time helpers.

## Contents

| File | Role |
|------|------|
| `../config.py` | Pydantic-settings wrapper over env vars (12-factor). |
| `db.py` | Postgres connection pool (`psycopg_pool`) + Redis client. |
| `logging.py` | structlog setup: console in dev, JSON in prod. |
| `time.py` | UTC-aware time helpers. |

## Conventions

- **Everything UTC.** No naive datetimes anywhere.
- **Settings are never mutated.** Use `settings()` (cached) and pass
  values into functions; don't stash module globals.
- **Pool sized small (1–10).** Tune upward only after profiling.
- **Dict-row cursors.** We return dicts everywhere so call sites can
  reference columns by name.
- **No ORM.** Raw SQL for perf-sensitive paths. If we later need an
  ORM for admin interfaces, use SQLAlchemy Core (not the declarative
  layer) to keep migrations simple.
