"""Unit tests for app.tools.ipca. The SGS API is simulated with httpx.MockTransport."""

from datetime import date

import httpx
import pytest

from app.tools.ipca import first_day_months_ago, get_ipca

TODAY = date(2026, 10, 25)


@pytest.mark.parametrize(
    ("today", "months", "expected"),
    [
        (date(2026, 10, 25), 0, date(2026, 10, 1)),
        (date(2026, 10, 25), 3, date(2026, 7, 1)),
        (date(2026, 2, 15), 3, date(2025, 11, 1)),  # crosses the year
        (date(2026, 1, 31), 12, date(2025, 1, 1)),
    ],
)
def test_first_day_months_ago(today, months, expected):
    assert first_day_months_ago(today, months) == expected


@pytest.mark.asyncio
async def test_returns_last_n_months_with_source():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json=[
                {"data": "01/07/2026", "valor": "0.07"},
                {"data": "01/08/2026", "valor": "-0.32"},
                {"data": "01/09/2026", "valor": "0.82"},
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await get_ipca(2, today=TODAY, client=client)

    # 2 months + 2 months of publication lag before October -> starts in June.
    assert captured[0].url.params["dataInicial"] == "01/06/2026"
    assert captured[0].url.params["dataFinal"] == "25/10/2026"
    assert result["sgs_code"] == 433
    assert result["retrieved_at"] == "2026-10-25"
    assert result["observations"] == [
        {"month": "2026-08", "value": "-0.32"},
        {"month": "2026-09", "value": "0.82"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("months", [0, 25, -1])
async def test_rejects_months_out_of_range(months):
    with pytest.raises(ValueError, match="months must be between"):
        await get_ipca(months, today=TODAY)
