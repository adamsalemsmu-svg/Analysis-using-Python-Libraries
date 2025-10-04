from __future__ import annotations
import os
from typing import Optional

import httpx


class NeyBotClient:
    """
    Thin client to interact with a natural‑language to SQL service ("ney bot").

    The client sends a question and optional schema hint to a configured endpoint
    and expects a JSON response containing the generated SQL query.

    Parameters
    ----------
    base_url : str, optional
        Base URL of the ney bot service. Defaults to the `NEY_BOT_BASE_URL`
        environment variable or ``http://localhost:8000`` if unset.
    api_key : str, optional
        API key to include in the Authorization header. Defaults to
        the `NEY_BOT_API_KEY` environment variable. If no key is provided,
        no Authorization header will be sent.
    timeout : float, optional
        Request timeout in seconds. Defaults to 20 seconds.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 20.0) -> None:
        self.base_url = base_url or os.getenv("NEY_BOT_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("NEY_BOT_API_KEY")
        self.timeout = timeout
        self._session = httpx.Client(timeout=timeout, headers=self._headers())

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def nl_to_sql(self, question: str, *, schema_hint: str | None = None) -> str:
        """
        Generate a SQL query from a natural‑language question using the ney bot service.

        Parameters
        ----------
        question : str
            The natural language question to translate into SQL.
        schema_hint : str, optional
            Additional hint describing the database schema to help the model generate
            more accurate queries (e.g. ``"Table iris(species, sepal_length, …)"``).

        Returns
        -------
        str
            The SQL query returned by the ney bot.

        Raises
        ------
        RuntimeError
            If the ney bot response does not contain a `sql` field or it's empty.
        httpx.HTTPStatusError
            If the HTTP request fails or returns a non‑success status code.
        """
        payload: dict[str, str] = {"question": question}
        if schema_hint:
            payload["schema"] = schema_hint

        response = self._session.post(f"{self.base_url}/generate_sql", json=payload)
        response.raise_for_status()
        data = response.json()
        sql: str = (data.get("sql") or "").strip()
        if not sql:
            raise RuntimeError("Ney bot returned empty SQL")
        return sql
