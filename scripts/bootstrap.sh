#!/usr/bin/env bash
set -euo pipefail

# End-to-end bootstrap for a fresh prototype.
# Assumes Docker is available and .env has been populated from .env.example.

cd "$(dirname "$0")/.."

echo "-- bringing up infra (postgres + redis + mlflow)"
docker compose up -d postgres redis mlflow

echo "-- waiting for postgres"
until docker compose exec -T postgres pg_isready -U "${PG_USER:-survpredict}" > /dev/null 2>&1; do
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
echo "  docker compose up -d inference warm-sweeper dashboard"
