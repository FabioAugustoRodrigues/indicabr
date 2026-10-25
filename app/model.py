"""Model client. Ollama exposes an OpenAI-compatible API, so the same code works for both."""

import os

from openai import AsyncOpenAI


def create_client() -> AsyncOpenAI:
    """Client configured from LLM_BASE_URL (and LLM_API_KEY, which Ollama ignores)."""
    return AsyncOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.getenv("LLM_API_KEY", "ollama"),
    )


def model_name() -> str:
    return os.environ["LLM_MODEL"]
