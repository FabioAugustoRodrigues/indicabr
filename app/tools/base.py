"""Common definition of a tool the agent can offer to the model."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    """A Python function plus the description the model reads to decide when to call it."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments
    function: Callable[..., Awaitable[dict[str, Any]]]

    @property
    def schema(self) -> dict[str, Any]:
        """The tool in the format expected by the OpenAI chat completions API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
