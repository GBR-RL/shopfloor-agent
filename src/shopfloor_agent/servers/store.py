"""Read and write access to the plant database, shared by the MCP servers.

Tools accept equipment by id ("CWC04006") or by name ("Chiller 6", "chiller6"), and date ranges
as ISO dates; an end date without a time includes that whole day.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError as _SdkToolError


class ToolError(_SdkToolError):
    """A problem with the tool arguments. The SDK returns its message to the agent as the tool
    result (other exceptions become a generic "Error executing tool")."""


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_bound(value: str | None, *, end: bool) -> str | None:
    """'2017' / '2017-03' / '2017-03-05' / full ISO -> ISO timestamp bound (end is exclusive)."""
    if value is None or not value.strip():
        return None
    v = value.strip()
    try:
        if re.fullmatch(r"\d{4}", v):
            d = date(int(v) + end, 1, 1)
        elif re.fullmatch(r"\d{4}-\d{2}", v):
            y, m = (int(p) for p in v.split("-"))
            d = date(y + (m == 12), m % 12 + 1, 1) if end else date(y, m, 1)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            d = date.fromisoformat(v) + timedelta(days=1 if end else 0)
        else:
            return datetime.fromisoformat(v).isoformat()
    except ValueError as exc:
        raise ToolError(f"invalid date '{value}': use YYYY, YYYY-MM or YYYY-MM-DD") from exc
    return datetime(d.year, d.month, d.day).isoformat()


class PlantStore:
    """One SQLite connection per store; a lock serialises the tool calls that share it."""

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} (build it with `shopfloor data`)")
        self.db_path = db_path
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._equipment = {
            r["equipment_id"]: dict(r) for r in self._db.execute("SELECT * FROM equipment")
        }
        self._names = {}
        for eq in self._equipment.values():
            self._names[_key(eq["equipment_id"])] = eq["equipment_id"]
            self._names[_key(eq["name"])] = eq["equipment_id"]

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._db

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._cursor() as db:
            return [dict(r) for r in db.execute(sql, params)]

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._cursor() as db:
            cur = db.execute(sql, params)
            db.commit()
            return cur.rowcount

    # -- lookups -------------------------------------------------------------------------
    def equipment_id(self, equipment: str) -> str:
        eq_id = self._names.get(_key(equipment))
        if eq_id is None:
            known = ", ".join(e["name"] for e in self.all_equipment())
            raise ToolError(f"unknown equipment '{equipment}'. Known: {known}")
        return str(eq_id)

    def equipment(self, equipment: str) -> dict[str, Any]:
        return dict(self._equipment[self.equipment_id(equipment)])

    def all_equipment(self) -> list[dict[str, Any]]:
        return [dict(e) for e in sorted(self._equipment.values(), key=_natural_name)]


def _natural_name(eq: dict[str, Any]) -> tuple[str, int]:
    m = re.match(r"(.*?)(\d+)$", eq["name"])
    return (m.group(1), int(m.group(2))) if m else (eq["name"], 0)
