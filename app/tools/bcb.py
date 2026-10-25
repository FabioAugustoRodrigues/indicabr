"""Client for the Banco Central do Brasil SGS API (Sistema Gerenciador de Séries Temporais)."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import httpx

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
TIMEOUT = 30.0  # seconds; long SGS queries can take 10-20s
SGS_DATE_FORMAT = "%d/%m/%Y"
NO_DATA_MARKER = "Value(s) not found"


@dataclass(frozen=True)
class Observation:
    """A single data point of an SGS series."""

    date: date
    value: Decimal


class SGSError(Exception):
    """Raised when the SGS API cannot be reached or returns an unusable response."""


async def fetch_sgs_series(
    code: int,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient | None = None,
) -> list[Observation]:
    """Fetch observations of SGS series `code` between `start_date` and `end_date` (inclusive)."""
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must not be after end_date ({end_date})"
        )

    # Without a client from the caller, open one and make sure it gets closed.
    if client is None:
        async with httpx.AsyncClient() as new_client:
            return await fetch_sgs_series(code, start_date, end_date, new_client)

    params = {
        "formato": "json",
        "dataInicial": start_date.strftime(SGS_DATE_FORMAT),
        # Always sent: omitting dataFinal can make the API answer 406.
        "dataFinal": end_date.strftime(SGS_DATE_FORMAT),
    }

    try:
        response = await client.get(
            SGS_URL.format(code=code), params=params, timeout=TIMEOUT
        )
        # SGS answers 404 (not an empty list) when the period has no data, e.g. a weekend
        # for a daily series. Other 404s (such as an unknown series code) are real errors.
        if response.status_code == 404 and NO_DATA_MARKER in response.text:
            return []
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as e:
        raise SGSError(f"SGS series {code}: request timed out after {TIMEOUT}s") from e
    except httpx.HTTPStatusError as e:
        raise SGSError(
            f"SGS series {code}: API returned HTTP {e.response.status_code}"
        ) from e
    except httpx.RequestError as e:
        raise SGSError(f"SGS series {code}: request failed ({e!r})") from e
    except ValueError as e:
        raise SGSError(f"SGS series {code}: response is not valid JSON") from e

    try:
        return [
            Observation(
                date=datetime.strptime(item["data"], SGS_DATE_FORMAT).date(),
                value=Decimal(item["valor"]),
            )
            for item in payload
        ]
    except (KeyError, TypeError, ValueError, InvalidOperation) as e:
        raise SGSError(f"SGS series {code}: unexpected response format") from e
