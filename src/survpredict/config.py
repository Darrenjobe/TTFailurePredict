"""Central configuration.

Pulled from environment variables (12-factor). Use ``settings()`` rather than
re-importing the module to get test-override friendliness.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NewRelicSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NR_", env_file=".env", extra="ignore")

    api_key: str = ""
    account_id: str = ""
    region: str = "US"
    ingest_license_key: str = ""

    @property
    def nerdgraph_url(self) -> str:
        return (
            "https://api.eu.newrelic.com/graphql"
            if self.region.upper() == "EU"
            else "https://api.newrelic.com/graphql"
        )

    @property
    def events_api_url(self) -> str:
        base = (
            "https://insights-collector.eu01.nr-data.net"
            if self.region.upper() == "EU"
            else "https://insights-collector.newrelic.com"
        )
        return f"{base}/v1/accounts/{self.account_id}/events"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = Field(default="dev", alias="SURVPREDICT_ENV")
    log_level: str = Field(default="INFO", alias="SURVPREDICT_LOG_LEVEL")

    pg_host: str = Field(default="localhost", alias="PG_HOST")
    pg_port: int = Field(default=5432, alias="PG_PORT")
    pg_db: str = Field(default="survpredict", alias="PG_DB")
    pg_user: str = Field(default="survpredict", alias="PG_USER")
    pg_password: str = Field(default="survpredict", alias="PG_PASSWORD")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-opus-4-7", alias="ANTHROPIC_MODEL")

    inference_host: str = Field(default="0.0.0.0", alias="INFERENCE_HOST")
    inference_port: int = Field(default=8080, alias="INFERENCE_PORT")

    propagation_hazard_threshold: float = Field(
        default=0.6, alias="PROPAGATION_HAZARD_THRESHOLD"
    )
    propagation_max_hops: int = Field(default=3, alias="PROPAGATION_MAX_HOPS")
    warm_sweep_seconds: int = Field(default=60, alias="WARM_SWEEP_SECONDS")

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def nr_settings() -> NewRelicSettings:
    return NewRelicSettings()
