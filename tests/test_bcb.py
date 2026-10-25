"""Unit tests for app.tools.bcb. The SGS API is simulated with httpx.MockTransport (no network)."""

from datetime import date
from decimal import Decimal

import httpx
import pytest

from app.tools.bcb import Observation, SGSError, fetch_sgs_series

pytestmark = pytest.mark.asyncio

START = date(2026, 7, 1)
END = date(2026, 9, 30)


def make_client(handler) -> httpx.AsyncClient:
    """Build a client whose requests are answered by `handler` instead of the real API."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_parses_observations():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"data": "01/08/2026", "valor": "-0.32"},
                {"data": "01/09/2026", "valor": "0.82"},
            ],
        )

    async with make_client(handler) as client:
        result = await fetch_sgs_series(433, START, END, client)

    assert result == [
        Observation(date=date(2026, 8, 1), value=Decimal("-0.32")),
        Observation(date=date(2026, 9, 1), value=Decimal("0.82")),
    ]


async def test_sends_series_code_and_dates_in_sgs_format():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=[])

    async with make_client(handler) as client:
        await fetch_sgs_series(433, START, END, client)

    request = captured[0]
    assert request.url.path == "/dados/serie/bcdata.sgs.433/dados"
    assert request.url.params["formato"] == "json"
    assert request.url.params["dataInicial"] == "01/07/2026"
    assert request.url.params["dataFinal"] == "30/09/2026"


async def test_returns_empty_list_when_period_has_no_data():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "erro": {
                    "statusCode": 404,
                    "detail": "SGSNegocioException: Value(s) not found",
                }
            },
        )

    async with make_client(handler) as client:
        assert await fetch_sgs_series(1, START, END, client) == []


async def test_other_404_raises_sgs_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="<html>Requisição inválida!</html>")

    async with make_client(handler) as client:
        with pytest.raises(SGSError, match="HTTP 404"):
            await fetch_sgs_series(99999999, START, END, client)


async def test_http_error_raises_sgs_error_with_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(406)

    async with make_client(handler) as client:
        with pytest.raises(SGSError, match="HTTP 406"):
            await fetch_sgs_series(1, START, END, client)


async def test_timeout_raises_sgs_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    async with make_client(handler) as client:
        with pytest.raises(SGSError, match="timed out"):
            await fetch_sgs_series(433, START, END, client)


async def test_network_failure_raises_sgs_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure", request=request)

    async with make_client(handler) as client:
        with pytest.raises(SGSError, match="request failed"):
            await fetch_sgs_series(433, START, END, client)


async def test_invalid_json_raises_sgs_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    async with make_client(handler) as client:
        with pytest.raises(SGSError, match="not valid JSON"):
            await fetch_sgs_series(433, START, END, client)


@pytest.mark.parametrize(
    "payload",
    [
        [{"data": "01/08/2026"}],  # missing "valor"
        [{"data": "2026-08-01", "valor": "0.1"}],  # wrong date format
        [{"data": "01/08/2026", "valor": "abc"}],  # value is not a number
        {"unexpected": "object"},  # not a list
    ],
)
async def test_unexpected_format_raises_sgs_error(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with make_client(handler) as client:
        with pytest.raises(SGSError, match="unexpected response format"):
            await fetch_sgs_series(433, START, END, client)


async def test_start_after_end_raises_value_error():
    with pytest.raises(ValueError):
        await fetch_sgs_series(433, END, START)
