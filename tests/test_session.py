"""Session storage behavior for PixeltableSession."""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime
from typing import Any

import pixeltable as pxt
import pytest
from agents.memory import Session, SessionSettings

from openai_agents_pixeltable import PixeltableSession


@pytest.fixture
def table_name() -> str:
    return f"tests.t{uuid.uuid4().hex}"


def _user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


@pytest.mark.asyncio
async def test_round_trip_preserves_order_and_detaches_items(table_name: str) -> None:
    session = PixeltableSession("conversation", table_name=table_name)
    assert isinstance(session, Session)

    await session.add_items([_user("one"), {"role": "user", "content": ("a", "b")}, _user("café 東京")])
    items = await session.get_items()
    assert items == [_user("one"), {"role": "user", "content": ["a", "b"]}, _user("café 東京")]

    items[0]["content"] = "changed"
    assert (await session.get_items())[0]["content"] == "one"

    rows = list(
        session.table.where(session.table.session_id == "conversation").select(session.table.created_at).collect()
    )
    assert len(rows) == 3
    assert all(isinstance(row["created_at"], datetime) and row["created_at"].tzinfo is not None for row in rows)
    indexes = session.table.get_metadata()["indexes"]
    assert "session_id_idx" in indexes
    assert "ordinal_idx" in indexes


@pytest.mark.asyncio
async def test_limit_semantics_match_session_adapter_contract(table_name: str) -> None:
    session = PixeltableSession("conversation", table_name=table_name, session_settings=SessionSettings(limit=2))
    await session.add_items([{"n": index} for index in range(5)])

    assert await session.get_items() == [{"n": 3}, {"n": 4}]
    assert await session.get_items(limit=None) == [{"n": 3}, {"n": 4}]
    assert await session.get_items(limit=4) == [{"n": index} for index in range(1, 5)]
    assert await session.get_items(limit=100) == [{"n": index} for index in range(5)]
    assert await session.get_items(limit=0) == []
    assert await session.get_items(limit=-1) == []

    unlimited = PixeltableSession("conversation", table_name=table_name)
    assert await unlimited.get_items() == [{"n": index} for index in range(5)]


@pytest.mark.asyncio
async def test_pop_clear_and_session_isolation(table_name: str) -> None:
    first = PixeltableSession("first", table_name=table_name)
    second = PixeltableSession("second", table_name=table_name)
    await first.add_items([{"n": 1}, {"n": 2}, {"n": 3}])
    await second.add_items([{"n": "keep"}])

    assert await first.pop_item() == {"n": 3}
    assert await first.pop_item() == {"n": 2}
    await first.add_items([{"n": 9}])
    assert await first.get_items() == [{"n": 1}, {"n": 9}]

    await first.clear_session()
    assert await first.get_items() == []
    assert await first.pop_item() is None
    assert await second.get_items() == [{"n": "keep"}]

    reopened = PixeltableSession("second", table_name=table_name)
    assert await reopened.get_items() == [{"n": "keep"}]


@pytest.mark.asyncio
async def test_empty_add_and_rejected_items_do_not_write(table_name: str) -> None:
    session = PixeltableSession("conversation", table_name=table_name)
    await session.add_items([])
    await session.add_items([_user("ok")])
    with pytest.raises(ValueError, match="JSON-serializable"):
        await session.add_items([_user("nope"), {"bad": {1, 2}}])  # type: ignore[dict-item]
    assert await session.get_items() == [_user("ok")]
    with pytest.raises(TypeError, match="list"):
        await session.add_items(_user("nope"))  # type: ignore[arg-type]


def test_settings_and_path_validation(table_name: str) -> None:
    settings = SessionSettings(limit=3)
    session = PixeltableSession("conversation", table_name=table_name, session_settings=settings)
    assert session.session_settings is settings

    coerced = PixeltableSession("conversation", table_name=table_name, session_settings={"limit": 1})
    assert coerced.session_settings == SessionSettings(limit=1)
    assert type(coerced.session_settings) is SessionSettings

    with pytest.raises(TypeError, match="Unknown session settings"):
        PixeltableSession("conversation", table_name=table_name, session_settings={"unknown": 1})
    with pytest.raises(TypeError, match="session_id"):
        PixeltableSession(123, table_name=table_name)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="version handle"):
        PixeltableSession("conversation", table_name="agents.sessions:3")
    with pytest.raises(ValueError, match="invalid Pixeltable table path"):
        PixeltableSession("conversation", table_name="agents..sessions")


@pytest.mark.asyncio
async def test_incompatible_table_is_rejected(table_name: str) -> None:
    parent = table_name.rsplit(".", 1)[0]
    pxt.create_dir(parent, if_exists="ignore")
    pxt.create_table(table_name, {"x": pxt.String})
    session = PixeltableSession("conversation", table_name=table_name)
    with pytest.raises(ValueError, match="cannot be used as an Agents session store"):
        await session.get_items()


@pytest.mark.asyncio
async def test_cancelled_add_still_persists(table_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    session = PixeltableSession("conversation", table_name=table_name)
    started = threading.Event()
    release = threading.Event()
    original = PixeltableSession._add_items_locked

    def delayed(self: PixeltableSession, payloads: list[Any]) -> None:
        started.set()
        assert release.wait(5)
        original(self, payloads)

    monkeypatch.setattr(PixeltableSession, "_add_items_locked", delayed)
    task = asyncio.create_task(session.add_items([_user("kept")]))
    # Wait from a worker thread so the event loop can run the mutation.
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await session.get_items() == [_user("kept")]
