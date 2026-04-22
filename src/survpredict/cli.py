"""Top-level CLI. ``survpredict <subcommand>``.

Thin wrapper over the module functions so an operator can drive everything
from the shell during prototype work.
"""

from __future__ import annotations

from pathlib import Path

import typer

from survpredict.common.logging import configure_logging, get_logger

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
log = get_logger(__name__)

ingest_app = typer.Typer()
feature_app = typer.Typer()
train_app = typer.Typer()
feedback_app = typer.Typer()
app.add_typer(ingest_app, name="ingest")
app.add_typer(feature_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(feedback_app, name="feedback")


@app.callback()
def _init() -> None:
    configure_logging()


# ---- ingest --------------------------------------------------------------

@ingest_app.command("nrql")
def ingest_nrql(entity_class: list[str] = typer.Option(default=None)):
    """Pull NRQL-driven features for the configured entity classes."""
    from survpredict.ingestion.nrql_puller import run_pull

    run_pull(entity_class)


@ingest_app.command("entities")
def ingest_entities(
    entity_type: str = typer.Argument(..., help="NR entityType, e.g. APPLICATION"),
    entity_class: str = typer.Option(
        None,
        "--entity-class",
        help="Override our class name (default: normalized from entity_type, e.g. APPLICATION -> apm.application)",
    ),
    skip_relationships: bool = typer.Option(False, help="Skip entity relationship snapshot"),
):
    from survpredict.ingestion.entity_graph import (
        normalize_entity_class,
        search_entities,
        snapshot_relationships,
        upsert_entities,
    )

    klass = entity_class or normalize_entity_class(entity_type)
    entities = search_entities(entity_type)
    upsert_entities(entities, entity_class=klass)
    edges = 0
    if not skip_relationships:
        try:
            edges = snapshot_relationships([e["guid"] for e in entities])
        except Exception as e:
            typer.echo(f"warning: relationship snapshot failed: {e}", err=True)
    typer.echo(f"upserted {len(entities)} entities as class={klass}, {edges} edges")


@ingest_app.command("backfill")
def ingest_backfill(
    days: int = typer.Option(7, help="How many days of history to backfill"),
    bucket_minutes: int = typer.Option(5, help="TIMESERIES bucket size in minutes"),
    entity_class: list[str] = typer.Option(None, "--entity-class", "-c"),
):
    """Pull historical NRQL TIMESERIES and materialize the features table."""
    from survpredict.ingestion.nrql_puller import run_backfill

    total = run_backfill(days=days, bucket_minutes=bucket_minutes, entity_classes=entity_class)
    typer.echo(f"backfilled {total} feature rows")


@ingest_app.command("postmortems")
def ingest_postmortems(directory: Path = typer.Argument(...)):
    from survpredict.ingestion.postmortem import ingest_markdown_dir

    n = ingest_markdown_dir(directory)
    typer.echo(f"ingested {n} new postmortems")


@ingest_app.command("incidents")
def ingest_incidents(
    days: int = typer.Option(30, "--days", "-d", help="Lookback window in days"),
    entity_type: str = typer.Option(
        "APPLICATION",
        "--entity-type",
        "-t",
        help="NR entityType filter. Pass '' to pull incidents for any entity type.",
    ),
    limit: int = typer.Option(5000, "--limit", help="NRQL LIMIT (NR max is 5000)"),
):
    """Pull NR alert incidents into the events table.

    Defaults to the last 30 days, APM applications only, LIMIT 5000.
    """
    from datetime import datetime, timedelta, timezone

    from survpredict.ingestion.events import pull_incidents

    since = datetime.now(timezone.utc) - timedelta(days=days)
    et = entity_type or None
    n = pull_incidents(since=since, entity_type=et, limit=limit)
    typer.echo(f"inserted {n} incidents (filter: days={days}, entity_type={et or 'any'}, limit={limit})")


# ---- features ------------------------------------------------------------

@feature_app.command("aggregate")
def feature_aggregate(entity_guid: str):
    from survpredict.features.aggregator import aggregate_for_entity

    out = aggregate_for_entity(entity_guid)
    typer.echo(f"wrote {len(out)} features for {entity_guid}")


@feature_app.command("list")
def feature_list():
    from survpredict.features.spec import load_feature_specs

    for s in load_feature_specs():
        typer.echo(f"{s.name:<30} classes={s.entity_classes} windows={s.windows}")


# ---- train ---------------------------------------------------------------

@train_app.command("run")
def train_run(
    entity_class: str = typer.Argument(...),
    lookback_days: int = typer.Option(90),
    horizon_min: int = typer.Option(60),
):
    from survpredict.training.pipeline import run_training

    result = run_training(entity_class, lookback_days=lookback_days, max_duration_minutes=horizon_min)
    typer.echo(result)


@train_app.command("promote")
def train_promote(version: str):
    from survpredict.training.registry import promote

    promote(version)
    typer.echo(f"promoted {version}")


# ---- feedback ------------------------------------------------------------

@feedback_app.command("structure")
def feedback_structure(limit: int = 10):
    from survpredict.feedback.structurer import structure_pending

    n = structure_pending(limit)
    typer.echo(f"structured {n} postmortems")


@feedback_app.command("reconcile")
def feedback_reconcile(lookback_hours: int = 168):
    from survpredict.feedback.reconcile import reconcile

    typer.echo(reconcile(lookback_hours=lookback_hours))


@feedback_app.command("propose")
def feedback_propose(event_id: str):
    from survpredict.feedback.proposer import propose_for_event

    pid = propose_for_event(event_id)
    typer.echo(f"proposal_id={pid}")


@feedback_app.command("retrain")
def feedback_retrain(entity_class: str, mode: str = "incremental"):
    from survpredict.feedback.retrain import full_retrain, incremental_retrain

    fn = full_retrain if mode == "full" else incremental_retrain
    typer.echo(fn(entity_class))


if __name__ == "__main__":
    app()
