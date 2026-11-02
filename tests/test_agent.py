"""Runner uses PixeltableSession across turns without a model API."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from agents import Agent, RunConfig, Runner, SessionSettings
from agents.testing import ScriptedModel, assistant_message

from examples.conversation import main as example_main
from openai_agents_pixeltable import PixeltableSession


def _run_sync(agent: Agent[Any], input_data: str, **kwargs: Any) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return Runner.run_sync(agent, input_data, **kwargs)
    finally:
        loop.close()


async def _run(method: str, agent: Agent[Any], input_data: str, **kwargs: Any) -> Any:
    """Drive Runner the way the SDK's session tests do."""
    if method == "run":
        return await Runner.run(agent, input_data, **kwargs)
    if method == "run_sync":
        return await asyncio.to_thread(_run_sync, agent, input_data, **kwargs)
    if method == "run_streamed":
        result = Runner.run_streamed(agent, input_data, **kwargs)
        async for _event in result.stream_events():
            pass
        return result
    raise AssertionError(method)


@pytest.fixture
def table_name() -> str:
    return f"tests.t{uuid.uuid4().hex}"


def _config(**kwargs: Any) -> RunConfig:
    return RunConfig(tracing_disabled=True, **kwargs)


def _texts(items: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            found.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    found.append(block["text"])
    return found


@pytest.mark.parametrize("runner_method", ["run", "run_sync", "run_streamed"])
@pytest.mark.asyncio
async def test_runner_two_turns_replay_stored_history(table_name: str, runner_method: str) -> None:
    session = PixeltableSession("bridge", table_name=table_name)
    model = ScriptedModel(
        [
            [assistant_message("San Francisco.")],
            [assistant_message("California.")],
        ]
    )
    agent = Agent(name="Assistant", instructions="Reply concisely.", model=model)
    config = _config()

    first = await _run(
        runner_method,
        agent,
        "What city is the Golden Gate Bridge in?",
        session=session,
        run_config=config,
    )
    assert first.final_output == "San Francisco."

    second = await _run(runner_method, agent, "What state is it in?", session=session, run_config=config)
    assert second.final_output == "California."

    model_input = model.calls[1].input
    assert isinstance(model_input, list)
    assert model_input[0]["content"] == "What city is the Golden Gate Bridge in?"
    assert model_input[-1]["content"] == "What state is it in?"
    assert len(model_input) > 1

    stored = await session.get_items()
    assert _texts(stored) == [
        "What city is the Golden Gate Bridge in?",
        "San Francisco.",
        "What state is it in?",
        "California.",
    ]

    other = PixeltableSession("other", table_name=table_name)
    assert await other.get_items() == []


@pytest.mark.asyncio
async def test_runner_session_settings_and_storage_contract(table_name: str) -> None:
    """Same contract as the SDK's offline custom-session example."""
    session = PixeltableSession("first", table_name=table_name, session_settings={"limit": 2})
    other = PixeltableSession("other", table_name=table_name)
    model = ScriptedModel(
        [
            [assistant_message("Hello!")],
            [assistant_message("You said hello.")],
        ]
    )
    agent = Agent(name="Assistant", model=model)
    config = _config()

    await Runner.run(agent, "Hello", session=session, run_config=config)
    await Runner.run(agent, "What did I say?", session=session, run_config=config)

    assert model.calls[-1].input[0]["content"] == "Hello"
    assert len(await session.get_items()) == 2
    assert len(await session.get_items(limit=4)) == 4
    assert await session.get_items(limit=0) == []
    assert await session.get_items(limit=-1) == []
    assert await other.get_items() == []

    last = (await session.get_items())[-1]
    assert await session.pop_item() == last
    await session.clear_session()
    assert await session.pop_item() is None
    assert await session.get_items() == []


@pytest.mark.parametrize("runner_method", ["run", "run_sync", "run_streamed"])
@pytest.mark.asyncio
async def test_run_config_limit_retrieves_newest_history(table_name: str, runner_method: str) -> None:
    session = PixeltableSession("limit", table_name=table_name)
    model = ScriptedModel()
    agent = Agent(name="test", model=model)
    config = _config()

    for index in range(1, 4):
        model.enqueue([assistant_message(f"Reply {index}")])
        await _run(runner_method, agent, f"Message {index}", session=session, run_config=config)

    assert len(await session.get_items()) == 6

    model.enqueue([assistant_message("Reply 4")])
    await _run(
        runner_method,
        agent,
        "Message 4",
        session=session,
        run_config=_config(session_settings=SessionSettings(limit=2)),
    )

    last_input = model.calls[-1].input
    assert len(last_input) == 3
    assert last_input[0]["content"] == "Message 3"
    assert last_input[1]["content"][0]["text"] == "Reply 3"
    assert last_input[2]["content"] == "Message 4"
    assert len(await session.get_items()) == 8


@pytest.mark.asyncio
async def test_separate_sessions_do_not_share_runner_history(table_name: str) -> None:
    model = ScriptedModel(
        [
            [assistant_message("I like cats")],
            [assistant_message("I don't remember")],
        ]
    )
    agent = Agent(name="test", model=model)
    config = _config()
    first = PixeltableSession("session-1", table_name=table_name)
    second = PixeltableSession("session-2", table_name=table_name)

    await Runner.run(agent, "I like cats", session=first, run_config=config)
    result = await Runner.run(agent, "Do you remember what I said?", session=second, run_config=config)

    assert result.final_output == "I don't remember"
    assert len(model.calls[-1].input) == 1
    assert _texts(await first.get_items()) == ["I like cats", "I like cats"]
    assert _texts(await second.get_items()) == ["Do you remember what I said?", "I don't remember"]


@pytest.mark.asyncio
async def test_example_runs_offline() -> None:
    assert await example_main() == "California."
    assert await example_main() == "California."
