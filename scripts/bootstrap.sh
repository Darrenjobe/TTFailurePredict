#!/usr/bin/env bash
set -euo pipefail

# End-to-end bootstrap for a fresh prototype.
# Assumes Docker is available and .env has been populated from .env.example.

cd "$(dirname "$0")/.."

# --- detect compose binary (v2 plugin vs standalone v1) ----------------------
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "error: neither 'docker compose' (v2) nor 'docker-compose' (v1) found." >&2
  echo "Install Docker Desktop or 'docker-compose', then re-run." >&2
  exit 1
fi
echo "-- using compose: ${COMPOSE[*]}"

# --- load .env so variables are available to this shell ----------------------
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "warning: .env not found; copy .env.example and fill in your keys." >&2
fi

PG_USER="${PG_USER:-survpredict}"

echo "-- bringing up infra (postgres + redis + mlflow)"
"${COMPOSE[@]}" up -d postgres redis mlflow

echo "-- waiting for postgres"
until "${COMPOSE[@]}" exec -T postgres pg_isready -U "$PG_USER" > /dev/null 2>&1; do
  sleep 1
done

echo "-- schema already applied via /docker-entrypoint-initdb.d"

# --- python venv -------------------------------------------------------------
# Homebrew pythons don't ship an unversioned `pip`, and recent releases mark
# the base env as externally-managed, so we always work inside .venv.
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found. brew install python@3.12 (or 3.11) and retry." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "-- creating .venv ($(python3 --version))"
  python3 -m venv .venv
fi

VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

echo "-- upgrading pip in .venv"
"$VENV_PY" -m pip install --upgrade pip >/dev/null

echo "-- installing survpredict into .venv (editable)"
"$VENV_PIP" install -e .

SURVPREDICT=".venv/bin/survpredict"

echo "-- syncing one entity class from NR (APPLICATION)"
"$SURVPREDICT" ingest entities APPLICATION

echo "-- pulling an initial batch of NRQL features"
"$SURVPREDICT" ingest nrql

echo "-- pulling recent incidents as event labels"
"$SURVPREDICT" ingest incidents

echo
echo "done. To use the CLI in this shell:"
echo "  source .venv/bin/activate"
echo
echo "Next:"
echo "  survpredict train run apm.application"
echo "  survpredict train promote <returned_version>"
echo "  ${COMPOSE[*]} up -d inference warm-sweeper dashboard"
