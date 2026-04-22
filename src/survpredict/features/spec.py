"""Feature spec loader.

Specs are declared as YAML documents in ``feature_specs/*.yaml`` (design doc
§6.2). Each spec becomes a compiled plan used by:

  - the NRQL puller (ingestion)
  - the feature aggregator (offline store)
  - the inference service (resolving current values at scoring time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC_DIR = REPO_ROOT / "feature_specs"

SourceKind = Literal["nr_metric", "nr_event", "derived"]
Transformation = Literal[
    "level",
    "trend_slope",
    "volatility_stddev",
    "volatility_mad",
    "changepoint_distance",
    "time_since_last_deploy",
    "cyclical_hour",
    "cyclical_dow",
]


@dataclass
class FeatureSpec:
    name: str
    entity_classes: list[str]
    source: SourceKind
    windows: list[int]
    transformations: list[Transformation] = field(default_factory=lambda: ["level"])
    nrql: str | None = None
    computation: str | None = None  # for derived features
    description: str | None = None

    def derived_column_names(self) -> list[str]:
        """Flattened column names produced by this spec after transformation."""
        names: list[str] = []
        for w in self.windows:
            for t in self.transformations:
                names.append(f"{self.name}__w{w}__{t}")
        return names


def load_feature_specs(spec_dir: Path | None = None) -> list[FeatureSpec]:
    spec_dir = Path(spec_dir) if spec_dir else DEFAULT_SPEC_DIR
    if not spec_dir.exists():
        return []
    specs: list[FeatureSpec] = []
    for path in sorted(spec_dir.glob("*.yaml")):
        docs = yaml.safe_load_all(path.read_text())
        for doc in docs:
            if doc is None:
                continue
            if isinstance(doc, list):
                for entry in doc:
                    specs.append(_from_dict(entry))
            else:
                specs.append(_from_dict(doc))
    _validate_unique_names(specs)
    return specs


def _from_dict(d: dict) -> FeatureSpec:
    return FeatureSpec(
        name=d["name"],
        entity_classes=list(d["entity_classes"]),
        source=d["source"],
        windows=list(d.get("windows") or [60]),
        transformations=list(d.get("transformations") or ["level"]),
        nrql=d.get("nrql"),
        computation=d.get("computation"),
        description=d.get("description"),
    )


def _validate_unique_names(specs: list[FeatureSpec]) -> None:
    seen: set[str] = set()
    for s in specs:
        if s.name in seen:
            raise ValueError(f"duplicate feature spec name: {s.name}")
        seen.add(s.name)
