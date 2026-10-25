"""Tests for the agent loop with a simulated model: no Ollama, no network."""

import copy
import json
from datetime import date
from types import SimpleNamespace

import pytest
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import Function

from app.agent import MAX_STEPS, STEP_LIMIT_ANSWER, run_agent
from app.tools.base import Tool
from app.tools.bcb import SGSError

pytestmark = pytest.mark.asyncio


class FakeModel:
    """Stands in for AsyncOpenAI: returns scripted replies and records what it was sent."""

    def __init__(self, replies: list[ChatCompletionMessage]):
        self.replies = list(replies)
        self.sent_messages: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        # Copy: the agent keeps appending to the same list after the call.
        self.sent_messages.append(copy.deepcopy(kwargs["messages"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


def text_reply(content: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=content)


def tool_call_reply(
    name: str, arguments: str, call_id: str = "call_1"
) -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id=call_id,
                type="function",
                function=Function(name=name, arguments=arguments),
            )
        ],
    )


def make_tool(function) -> Tool:
    return Tool(
        name="get_ipca",
        description="test tool",
        parameters={"type": "object", "properties": {"months": {"type": "integer"}}},
        function=function,
    )


async def fake_ipca(months: int) -> dict:
    return {"sgs_code": 433, "observations": [{"month": "2026-09", "value": "0.82"}]}


def tool_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["role"] == "tool"]


async def test_answers_directly_when_no_tool_is_needed():
    model = FakeModel([text_reply("Olá!")])

    answer = await run_agent("oi", model, "fake", tools=[make_tool(fake_ipca)])

    assert answer == "Olá!"
    assert len(model.sent_messages) == 1


async def test_system_prompt_includes_today():
    model = FakeModel([text_reply("ok")])

    await run_agent("oi", model, "fake", today=date(2026, 10, 25))

    assert "25/10/2026" in model.sent_messages[0][0]["content"]


async def test_runs_requested_tool_and_sends_result_back():
    model = FakeModel(
        [tool_call_reply("get_ipca", '{"months": 1}'), text_reply("IPCA: 0,82%")]
    )

    answer = await run_agent("IPCA?", model, "fake", tools=[make_tool(fake_ipca)])

    assert answer == "IPCA: 0,82%"
    second_call = model.sent_messages[1]
    # History sent back: system, user, the model's tool call, then the tool result.
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "tool"]
    [tool_message] = tool_messages(second_call)
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["observations"][0]["value"] == "0.82"


async def test_tool_failure_is_returned_to_model_as_error():
    async def failing_ipca(months: int) -> dict:
        raise SGSError("SGS series 433: request timed out after 30.0s")

    model = FakeModel(
        [tool_call_reply("get_ipca", '{"months": 1}'), text_reply("Fonte indisponível")]
    )

    answer = await run_agent("IPCA?", model, "fake", tools=[make_tool(failing_ipca)])

    assert answer == "Fonte indisponível"
    [tool_message] = tool_messages(model.sent_messages[1])
    assert "timed out" in json.loads(tool_message["content"])["error"]


@pytest.mark.parametrize(
    ("name", "arguments", "expected_error"),
    [
        ("get_ipca", "{not json", "JSONDecodeError"),
        ("get_ipca", '["a list"]', "must be a JSON object"),
        ("get_ipca", '{"wrong_arg": 1}', "TypeError"),
        ("get_selic", "{}", "unknown tool"),
    ],
)
async def test_bad_tool_calls_are_returned_to_model_as_error(
    name, arguments, expected_error
):
    model = FakeModel([tool_call_reply(name, arguments), text_reply("desculpe")])

    await run_agent("IPCA?", model, "fake", tools=[make_tool(fake_ipca)])

    [tool_message] = tool_messages(model.sent_messages[1])
    assert expected_error in json.loads(tool_message["content"])["error"]


async def test_stops_after_max_steps():
    # A model stuck asking for tools forever.
    model = FakeModel(
        [tool_call_reply("get_ipca", '{"months": 1}') for _ in range(MAX_STEPS + 1)]
    )

    answer = await run_agent("IPCA?", model, "fake", tools=[make_tool(fake_ipca)])

    assert answer == STEP_LIMIT_ANSWER
    assert len(model.sent_messages) == MAX_STEPS
