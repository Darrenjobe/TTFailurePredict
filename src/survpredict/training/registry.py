"""MLflow-backed model registry.

Logs trained artifacts and writes a summary into ``model_registry_meta`` so
that the inference service can find the current production version per class
without hitting MLflow on the hot path.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.config import settings

log = get_logger(__name__)


def _configure_mlflow() -> None:
    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    mlflow.set_experiment("survpredict")


def log_model(
    trained,
    report,
    algorithm: str,
    extra_tags: dict | None = None,
) -> str:
    """Log artifact to MLflow and mirror metadata to Postgres. Returns model_version."""
    _configure_mlflow()
    model_version = f"{trained.entity_class}:{algorithm}:{int(datetime.now(timezone.utc).timestamp())}"

    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = Path(tmp) / "model.pkl"
        with artifact_path.open("wb") as f:
            pickle.dump({
                "trained": trained,
                "feature_columns": trained.feature_columns,
            }, f)

        with mlflow.start_run(run_name=model_version):
            mlflow.log_params({
                "entity_class": trained.entity_class,
                "algorithm": algorithm,
                "training_size": trained.training_size,
                **(trained.params if hasattr(trained, "params") else {}),
            })
            mlflow.log_metrics({
                "c_index": report.c_index,
                "ibs": report.ibs or 0.0,
                "event_rate": report.event_rate,
                **{f"time_auc_{h}": v for h, v in (report.time_auc or {}).items()},
            })
            mlflow.log_artifact(str(artifact_path))
            if extra_tags:
                mlflow.set_tags(extra_tags)
            run = mlflow.active_run()
            artifact_uri = f"{run.info.artifact_uri}/{artifact_path.name}"

    _mirror_to_db(model_version, trained, report, algorithm, artifact_uri)
    log.info("model_logged", version=model_version, c_index=report.c_index)
    return model_version


def _mirror_to_db(model_version, trained, report, algorithm, artifact_uri) -> None:
    with pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_registry_meta (model_version, entity_class, algorithm,
                c_index, ibs, trained_at, artifact_uri, feature_list, is_production)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (model_version) DO UPDATE SET
                c_index = EXCLUDED.c_index,
                ibs = EXCLUDED.ibs,
                artifact_uri = EXCLUDED.artifact_uri
            """,
            (
                model_version,
                trained.entity_class,
                algorithm,
                report.c_index,
                report.ibs,
                trained.trained_at,
                artifact_uri,
                json.dumps(trained.feature_columns),
            ),
        )


def promote(model_version: str) -> None:
    """Mark a model as the production model for its entity_class (§7.5 canary promotion)."""
    with pg_cursor() as cur:
        cur.execute(
            "SELECT entity_class FROM model_registry_meta WHERE model_version = %s",
            (model_version,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"unknown model_version: {model_version}")
        klass = row["entity_class"]
        cur.execute(
            "UPDATE model_registry_meta SET is_production = FALSE WHERE entity_class = %s",
            (klass,),
        )
        cur.execute(
            """
            UPDATE model_registry_meta
            SET is_production = TRUE, promoted_at = NOW()
            WHERE model_version = %s
            """,
            (model_version,),
        )
    log.info("model_promoted", version=model_version, entity_class=klass)


def load_production(entity_class: str):
    """Load the production model for an entity class into memory."""
    _configure_mlflow()
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT model_version, artifact_uri, feature_list
            FROM model_registry_meta
            WHERE entity_class = %s AND is_production = TRUE
            ORDER BY promoted_at DESC
            LIMIT 1
            """,
            (entity_class,),
        )
        row = cur.fetchone()
    if not row:
        return None
    local = mlflow.artifacts.download_artifacts(row["artifact_uri"])
    with open(local, "rb") as f:
        return {
            "version": row["model_version"],
            **pickle.load(f),
        }
