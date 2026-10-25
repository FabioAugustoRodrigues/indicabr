"""Command-line chat with the agent.

Usage: uv run --env-file .env python scripts/ask.py ["question"]
Without a question, starts an interactive prompt (empty line or "sair" to quit).
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import run_agent
from app.model import create_client, model_name


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  [%(name)s] %(message)s")
    client = create_client()
    model = model_name()

    if len(sys.argv) > 1:
        print(await run_agent(" ".join(sys.argv[1:]), client, model))
        return

    while True:
        question = input("\nPergunta: ").strip()
        if question.lower() in ("", "sair"):
            break
        print(await run_agent(question, client, model))


asyncio.run(main())
