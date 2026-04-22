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
def ingest_entities(entity_type: str = typer.Argument(..., help="NR entityType, e.g. APPLICATION")):
    from survpredict.ingestion.entity_graph import search_entities, snapshot_relationships, upsert_entities

    entities = search_entities(entity_type)
    upsert_entities(entities, entity_class=entity_type.lower())
    snapshot_relationships([e["guid"] for e in entities])
    typer.echo(f"upserted {len(entities)} entities")


@ingest_app.command("postmortems")
def ingest_postmortems(directory: Path = typer.Argument(...)):
    from survpredict.ingestion.postmortem import ingest_markdown_dir

    n = ingest_markdown_dir(directory)
    typer.echo(f"ingested {n} new postmortems")


@ingest_app.command("incidents")
def ingest_incidents():
    from survpredict.ingestion.events import pull_incidents

    n = pull_incidents()
    typer.echo(f"pulled {n} incidents")


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
