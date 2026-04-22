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

echo "-- installing python package"
pip install -e .

echo "-- syncing one entity class from NR (APPLICATION)"
survpredict ingest entities APPLICATION

echo "-- pulling an initial batch of NRQL features"
survpredict ingest nrql

echo "-- pulling recent incidents as event labels"
survpredict ingest incidents

echo "done. Next:"
echo "  survpredict train run apm.application"
echo "  survpredict train promote <returned_version>"
echo "  ${COMPOSE[*]} up -d inference warm-sweeper dashboard"
