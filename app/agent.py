"""The agent loop: ask the model, run the tools it requests, repeat until it answers."""

import json
import logging
import time
from collections.abc import Sequence
from datetime import date
from typing import Any

from openai import AsyncOpenAI

from app.tools.base import Tool
from app.tools.ipca import IPCA_TOOL

logger = logging.getLogger(__name__)

MAX_STEPS = 5
DEFAULT_TOOLS: tuple[Tool, ...] = (IPCA_TOOL,)
STEP_LIMIT_ANSWER = "Não consegui chegar a uma resposta dentro do limite de passos."

# Written in Portuguese: small models answer more reliably in the prompt's language.
SYSTEM_PROMPT = """\
Você é o IndicaBR, um assistente que responde perguntas sobre indicadores econômicos do Brasil.
Data de hoje: {today}.

Regras:
- Use somente dados retornados pelas ferramentas. Nunca invente nem estime números.
- Não faça contas: informe os valores exatamente como vieram das ferramentas.
- Sempre cite a fonte: nome da série, código SGS, período dos dados e data da consulta.
- Se uma ferramenta retornar erro, explique o problema ao usuário sem inventar uma resposta.
- Se a pergunta não puder ser respondida com as ferramentas disponíveis, diga isso.
- Responda em português do Brasil, de forma curta e direta.
"""


async def run_agent(
    question: str,
    client: AsyncOpenAI,
    model: str,
    tools: Sequence[Tool] = DEFAULT_TOOLS,
    today: date | None = None,
) -> str:
    """Answer `question`, letting the model call `tools` up to MAX_STEPS times."""
    today = today or date.today()
    tools_by_name = {tool.name: tool for tool in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=f"{today:%d/%m/%Y}")},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_STEPS):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool.schema for tool in tools],
        )
        reply = response.choices[0].message
        # The model keeps no memory between calls: its own reply (including the tool
        # calls it made) must go back in the history so it can link the results.
        messages.append(reply.model_dump(exclude_none=True))

        if not reply.tool_calls:
            return reply.content or ""

        for call in reply.tool_calls:
            result = await run_tool(
                tools_by_name, call.function.name, call.function.arguments
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    logger.warning("step limit reached (%d) for question: %r", MAX_STEPS, question)
    return STEP_LIMIT_ANSWER


async def run_tool(
    tools_by_name: dict[str, Tool], name: str, raw_arguments: str
) -> dict[str, Any]:
    """Run one tool call. Any failure is returned as {"error": ...} for the model to handle."""
    started = time.perf_counter()
    try:
        tool = tools_by_name.get(name)
        if tool is None:
            raise ValueError(f"unknown tool: {name}")
        arguments = json.loads(raw_arguments or "{}")
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a JSON object")
        result = await tool.function(**arguments)
    except Exception as e:
        # Deliberately broad: a failing tool must not crash the agent. The error goes
        # back to the model, and the traceback goes to the log for debugging.
        logger.exception(
            "tool=%s args=%s failed in %.0fms",
            name,
            raw_arguments,
            (time.perf_counter() - started) * 1000,
        )
        return {"error": f"{type(e).__name__}: {e}"}

    logger.info(
        "tool=%s args=%s ok in %.0fms",
        name,
        raw_arguments,
        (time.perf_counter() - started) * 1000,
    )
    return result
