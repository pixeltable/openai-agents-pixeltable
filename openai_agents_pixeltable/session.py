"""Pixeltable backend for an OpenAI Agents SDK ``Session``."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

import pixeltable as pxt
from agents import SessionSettings, TResponseInputItem
from agents.memory import await_session_mutation, coerce_session_settings, resolve_session_limit
from pixeltable.catalog import Path

_TABLE_COMMENT = "Conversation history for the OpenAI Agents SDK. Managed by openai-agents-pixeltable."
_SESSION_INDEX = "session_id_idx"

# Stored columns this package owns. item_id is the primary key.
# Other columns may exist; the session does not read or write them.
_COLUMNS: dict[str, str] = {
    "item_id": "String",
    "session_id": "String",
    "ordinal": "Int",
    "item": "Json",
    "created_at": "Timestamp",
}


def _schema() -> dict[str, Any]:
    return {
        "item_id": pxt.String,
        "session_id": pxt.String,
        "ordinal": pxt.Int,
        "item": pxt.Json,
        "created_at": pxt.Timestamp,
    }


def _column_base(type_: object) -> str:
    return str(type_).split(" | ", 1)[0].split("[", 1)[0]


def _is_nullable(type_: object) -> bool:
    return str(type_).endswith(" | None")


def _table_path(table_name: str) -> Path:
    """Parse ``table_name`` with Pixeltable's path rules and reject a version handle."""
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError(f"invalid Pixeltable table path: {table_name!r}")
    try:
        path = Path.parse(table_name, allow_versioned_path=True)
    except pxt.Error as exc:
        raise ValueError(f"invalid Pixeltable table path: {table_name!r}") from exc
    if path.version is not None:
        raise ValueError(f"table path {table_name!r} must not be a version handle")
    if not path.components:
        raise ValueError(f"invalid Pixeltable table path: {table_name!r}")
    return path


def _has_nul(value: Any) -> bool:
    """True when a JSON value contains a NUL character. Postgres JSONB rejects those."""
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(key) or _has_nul(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_has_nul(child) for child in value)
    return False


def _encode_item(item: TResponseInputItem) -> Any:
    """Snapshot an item as JSON the store can accept. NaN, Infinity, and NUL are rejected."""
    try:
        encoded = json.dumps(item, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("session item must be JSON-serializable") from exc
    decoded = json.loads(encoded)
    if _has_nul(decoded):
        raise ValueError("session item must not contain a NUL character")
    return decoded


class PixeltableSession:
    """OpenAI Agents ``Session`` persisted in one Pixeltable table.

    Rows are conversation items. ``session_id`` selects one conversation.
    ``ordinal`` is the insertion order inside that conversation, starting at 1.
    ``item`` is the JSON item the SDK stores. ``created_at`` is the UTC write time.
    ``item_id`` is the primary key.

    The table is created on first use. A btree index is added on ``session_id``.
    Reopening a compatible table does not recreate it. Columns this package does
    not own are left in place. A missing table is created again on the next call.

    ``get_items`` uses :func:`agents.memory.resolve_session_limit`. An explicit
    ``limit`` wins. Otherwise ``session_settings.limit`` is used. ``None`` reads
    the whole session, oldest first. A positive limit returns that many newest
    items, still oldest-first. Zero and negative limits return no items.

    Writes go through :func:`agents.memory.await_session_mutation`, so cancelling
    the caller waits for the write to finish and does not roll it back. The
    in-process lock serializes operations on one table. Dotted, slash, and
    ``pxt://`` spellings of that table share the lock, and letter case does not
    matter. Overlapping runs that share a session from separate processes still
    need application coordination.

    Args:
        session_id: Conversation id stored on every row this instance reads or writes.
        table_name: Pixeltable table path, for example ``agents.sessions``,
            ``agents/sessions``, or ``pxt://org:db/agents/sessions``.
        session_settings: Default retrieval settings. A dict is coerced with
            :func:`agents.memory.coerce_session_settings`. ``None`` uses
            ``SessionSettings()``.
    """

    _locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_guard: ClassVar[threading.Lock] = threading.Lock()

    session_settings: SessionSettings

    def __init__(
        self,
        session_id: str,
        *,
        table_name: str = "agents.sessions",
        session_settings: SessionSettings | dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        path = _table_path(table_name)

        self.session_id = session_id
        self.table_name = table_name
        self._path = path
        self.session_settings = (
            SessionSettings() if session_settings is None else coerce_session_settings(session_settings)
        )
        self._table: pxt.Table | None = None

    def __repr__(self) -> str:
        return f"PixeltableSession(session_id={self.session_id!r}, table_name={self.table_name!r})"

    @property
    def table(self) -> pxt.Table:
        """The Pixeltable table. Available for queries this session does not wrap."""
        return self._locked(self._ensure_table)

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return conversation history, oldest first.

        ``limit`` follows :func:`agents.memory.resolve_session_limit`. ``None``
        uses ``session_settings.limit``. A positive value is the newest N items.
        Zero or negative returns an empty list.
        """
        resolved = resolve_session_limit(limit, self.session_settings)
        return await asyncio.to_thread(self._locked, self._get_items_locked, resolved)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append items in order. An empty list does not write."""
        if not isinstance(items, list):
            raise TypeError("items must be a list")
        if not items:
            return
        payloads = [_encode_item(item) for item in items]
        await await_session_mutation(asyncio.to_thread(self._locked, self._add_items_locked, payloads))

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the newest item, or ``None`` when the session is empty."""
        return await await_session_mutation(asyncio.to_thread(self._locked, self._pop_item_locked))

    async def clear_session(self) -> None:
        """Delete every item for this ``session_id``. Other sessions in the table stay."""
        await await_session_mutation(asyncio.to_thread(self._locked, self._clear_locked))

    @classmethod
    def _lock_for(cls, table_key: str) -> threading.Lock:
        with cls._locks_guard:
            lock = cls._locks.get(table_key)
            if lock is None:
                lock = threading.Lock()
                cls._locks[table_key] = lock
            return lock

    def _locked(self, fn: Any, *args: Any) -> Any:
        with self._lock_for(str(self._path)):
            try:
                return fn(*args)
            except pxt.NotFoundError:
                # The cached handle belongs to a table that was dropped.
                self._table = None
                return fn(*args)

    def _ensure_dirs(self) -> None:
        parent = self._path.parent
        if parent.is_root:
            return
        pxt.create_dir(str(parent), if_exists="ignore", parents=True)

    def _ensure_table(self) -> pxt.Table:
        if self._table is not None:
            return self._table
        path = str(self._path)
        table = pxt.get_table(path, if_not_exists="ignore")
        if table is None:
            self._ensure_dirs()
            table = pxt.create_table(
                path,
                _schema(),
                primary_key="item_id",
                comment=_TABLE_COMMENT,
                custom_metadata={"component": "openai-agents-pixeltable", "role": "session"},
                if_exists="ignore",
            )
        self._check_schema(table)
        self._ensure_indexes(table)
        self._table = table
        return table

    def _check_schema(self, table: pxt.Table) -> None:
        metadata = table.get_metadata()
        problems: list[str] = []
        if metadata.get("kind") != "table":
            problems.append(f"kind is {metadata.get('kind')!r}, expected 'table'")
        primary_key = list(metadata.get("primary_key") or [])
        if primary_key != ["item_id"]:
            problems.append(f"primary key is {primary_key}, expected ['item_id']")
        columns = metadata.get("columns") or {}
        missing = sorted(name for name in _COLUMNS if name not in columns)
        if missing:
            problems.append(f"missing columns {missing}")
        for name, expected in _COLUMNS.items():
            info = columns.get(name)
            if info is None:
                continue
            if info.get("is_computed"):
                problems.append(f"column {name!r} must be stored, not computed")
            actual = info.get("type_")
            if _column_base(actual) != expected:
                problems.append(f"column {name!r} has type {actual!r}, expected {expected}")
            if _is_nullable(actual):
                problems.append(f"column {name!r} must be required")
        if problems:
            detail = "; ".join(problems)
            raise ValueError(
                f"Pixeltable table {self.table_name!r} cannot be used as an Agents session store: {detail}"
            )

    def _ensure_indexes(self, table: pxt.Table) -> None:
        indexes = table.get_metadata().get("indexes") or {}
        if _SESSION_INDEX not in indexes:
            table.add_btree_index("session_id", idx_name=_SESSION_INDEX, if_exists="ignore")

    def _session_query(self, table: pxt.Table, *, asc: bool) -> Any:
        return table.where(table.session_id == self.session_id).order_by(
            table.ordinal, table.created_at, table.item_id, asc=asc
        )

    def _get_items_locked(self, limit: int | None) -> list[TResponseInputItem]:
        if limit is not None and limit <= 0:
            self._ensure_table()
            return []
        table = self._ensure_table()
        if limit is None:
            rows = list(self._session_query(table, asc=True).select(table.item).collect())
            return [dict(row)["item"] for row in rows]
        rows = list(self._session_query(table, asc=False).limit(limit).select(table.item).collect())
        return [dict(row)["item"] for row in reversed(rows)]

    def _next_ordinal(self, table: pxt.Table) -> int:
        rows = list(
            table.where(table.session_id == self.session_id)
            .order_by(table.ordinal, asc=False)
            .limit(1)
            .select(table.ordinal)
            .collect()
        )
        if not rows:
            return 1
        return int(dict(rows[0])["ordinal"]) + 1

    def _add_items_locked(self, payloads: list[Any]) -> None:
        table = self._ensure_table()
        start = self._next_ordinal(table)
        created_at = datetime.now(UTC)
        rows = [
            {
                "item_id": uuid.uuid4().hex,
                "session_id": self.session_id,
                "ordinal": start + offset,
                "item": payload,
                "created_at": created_at,
            }
            for offset, payload in enumerate(payloads)
        ]
        table.insert(rows)

    def _pop_item_locked(self) -> TResponseInputItem | None:
        table = self._ensure_table()
        rows = list(self._session_query(table, asc=False).limit(1).select(table.item_id, table.item).collect())
        if not rows:
            return None
        row = dict(rows[0])
        table.delete(table.item_id == row["item_id"])
        return row["item"]

    def _clear_locked(self) -> None:
        table = self._ensure_table()
        table.delete(table.session_id == self.session_id)
