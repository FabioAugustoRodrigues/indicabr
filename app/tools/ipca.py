"""IPCA tool: monthly inflation from the Banco Central SGS (series 433)."""

from datetime import date
from typing import Any

import httpx

from app.tools.base import Tool
from app.tools.bcb import SGS_URL, fetch_sgs_series

IPCA_SERIES = 433
MAX_MONTHS = 24
# IPCA is published around the 10th of the following month, so the latest value is
# one or two months behind today. Fetch a wider window and keep the last N values.
PUBLICATION_LAG_MONTHS = 2


def first_day_months_ago(today: date, months: int) -> date:
    """First day of the month that is `months` months before `today`'s month."""
    index = today.year * 12 + (today.month - 1) - months
    return date(index // 12, index % 12 + 1, 1)


async def get_ipca(
    months: int,
    today: date | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Return the last `months` published monthly IPCA values, with source metadata."""
    if not 1 <= months <= MAX_MONTHS:
        raise ValueError(f"months must be between 1 and {MAX_MONTHS}, got {months}")

    today = today or date.today()
    start = first_day_months_ago(today, months + PUBLICATION_LAG_MONTHS)
    observations = await fetch_sgs_series(IPCA_SERIES, start, today, client)
    latest = observations[-months:]

    return {
        "series": "IPCA - variação mensal (%)",
        "sgs_code": IPCA_SERIES,
        "source": "Banco Central do Brasil - SGS",
        "source_url": SGS_URL.format(code=IPCA_SERIES),
        "retrieved_at": today.isoformat(),
        # Values stay as strings so no precision is lost on the way to the model.
        "observations": [
            {"month": f"{obs.date:%Y-%m}", "value": str(obs.value)} for obs in latest
        ],
    }


IPCA_TOOL = Tool(
    name="get_ipca",
    description=(
        "Returns the monthly IPCA (official Brazilian consumer inflation, % change "
        "per month) for the most recent published months. Use it for questions about "
        "monthly inflation or the IPCA of a given recent month."
    ),
    parameters={
        "type": "object",
        "properties": {
            "months": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_MONTHS,
                "description": "How many of the most recent months to return (1 = only the latest).",
            }
        },
        "required": ["months"],
    },
    function=get_ipca,
)
