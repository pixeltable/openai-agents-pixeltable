# openai-agents-pixeltable

[![CI](https://github.com/pixeltable/openai-agents-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/openai-agents-pixeltable/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openai-agents-pixeltable.svg)](https://pypi.org/project/openai-agents-pixeltable/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/pixeltable/openai-agents-pixeltable/blob/main/LICENSE)

Pixeltable session storage for the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). `PixeltableSession` implements the SDK [`Session`](https://openai.github.io/openai-agents-python/sessions/) protocol, so `Runner` loads and saves conversation history in a Pixeltable table ([docs](https://docs.pixeltable.com), [source](https://github.com/pixeltable/pixeltable)).

Requires Python 3.11+ (Pixeltable's floor) and Pixeltable 0.7.22+.

The session calls the public adapter helpers added in [openai/openai-agents-python#5428](https://github.com/openai/openai-agents-python/pull/5428) (`await_session_mutation`, `coerce_session_settings`, `resolve_session_limit`). That change is not in the PyPI `openai-agents` 0.24.0 release. This package depends on the tested git commit that contains it.

## Install

```bash
pip install openai-agents-pixeltable
```

Set `PIXELTABLE_HOME` before importing this package if the catalog should not live in the default Pixeltable directory. The variable is read when Pixeltable initializes.

## Usage

```python
import asyncio

from agents import Agent, Runner

from openai_agents_pixeltable import PixeltableSession


async def main() -> None:
    agent = Agent(
        name="Assistant",
        instructions="Reply concisely.",
    )
    session = PixeltableSession("conversation-123")

    result = await Runner.run(
        agent,
        "What city is the Golden Gate Bridge in?",
        session=session,
    )
    print(result.final_output)

    result = await Runner.run(agent, "What state is it in?", session=session)
    print(result.final_output)


asyncio.run(main())
```

[`examples/conversation.py`](examples/conversation.py) runs the same two turns with a scripted model and does not need an API key.

## Configuration

```python
from agents import SessionSettings

from openai_agents_pixeltable import PixeltableSession

session = PixeltableSession(
    "conversation-123",
    table_name="agents.sessions",
    session_settings=SessionSettings(limit=50),
)
```

`table_name` is the Pixeltable path of the shared store. Every `session_id` is a set of rows in that table. The table is created on first use.

| Column | Type | Role |
| --- | --- | --- |
| `item_id` | string, primary key | row id |
| `session_id` | string | conversation id, indexed |
| `ordinal` | int | insertion order within the session, indexed |
| `item` | json | one history item |
| `created_at` | timestamp | UTC write time |

`get_items(limit)` follows the SDK adapter rules:

- no explicit limit: use `session_settings.limit`
- limit is `None` on the settings too: return every item, oldest first
- positive limit: return that many newest items, oldest first
- zero or negative limit: return no items

`RunConfig.session_settings` overrides the session default for one run. `add_items`, `pop_item`, and `clear_session` wait for the write to finish if the caller is cancelled, and `clear_session` deletes only that `session_id`.

Operations on one table are serialized in-process. Coordinate overlapping runs yourself when several processes write the same session. `session.table` is the Pixeltable table, for queries this class does not wrap.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check . && ruff format --check .
mypy openai_agents_pixeltable
```

Tests use a Pixeltable catalog under a temp directory and a scripted model. They do not need an API key.

## License

Apache 2.0
