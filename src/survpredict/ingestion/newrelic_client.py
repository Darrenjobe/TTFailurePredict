"""Thin NerdGraph (GraphQL) client.

Handles auth, retry with backoff, and shallow error translation. Higher-level
modules compose queries on top of this and do not talk to httpx directly.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from survpredict.common.logging import get_logger
from survpredict.config import nr_settings

log = get_logger(__name__)


class NewRelicError(RuntimeError):
    pass


class NewRelicClient:
    def __init__(self, timeout: float = 30.0):
        s = nr_settings()
        if not s.api_key:
            raise NewRelicError("NR_API_KEY is not configured")
        self._url = s.nerdgraph_url
        self._headers = {
            "API-Key": s.api_key,
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NewRelicClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"query": query, "variables": variables or {}}
        resp = self._client.post(self._url, json=payload, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise NewRelicError(f"NerdGraph errors: {data['errors']}")
        return data["data"]

    def nrql(self, nrql_query: str) -> list[dict[str, Any]]:
        """Run an NRQL query via NerdGraph. Returns the `results` list."""
        s = nr_settings()
        q = """
        query($accountId: Int!, $nrql: Nrql!) {
          actor {
            account(id: $accountId) {
              nrql(query: $nrql) {
                results
                metadata { eventTypes }
              }
            }
          }
        }
        """
        variables = {"accountId": int(s.account_id), "nrql": nrql_query}
        data = self.graphql(q, variables)
        return data["actor"]["account"]["nrql"]["results"]
