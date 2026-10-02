"""Maintenance MCP server: work orders, events and alerts.

Read tools are always exposed. Write tools (create / update / close / cancel a work order) are a
separate tier: `read_only=True` leaves them out of the server entirely, so a deployment can give
an agent no way to change records rather than relying on the agent to refrain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from shopfloor_agent.servers.store import PlantStore, ToolError, parse_bound

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

WorkType = Literal["PM", "CM"]  # preventive / corrective
GroupBy = Literal["work_type", "component", "primary_code", "year", "month", "status"]
OPEN_STATUSES = ("WAPPR", "APPR", "INPRG")
STATUSES = ("WAPPR", "APPR", "INPRG", "COMP", "CLOSE", "CAN")
MAX_ROWS = 50

_GROUP_SQL = {
    "work_type": "work_type",
    "component": "component",
    "primary_code": "COALESCE(primary_code, '')",
    "year": "substr(COALESCE(finished_at, reported_at), 1, 4)",
    "month": "substr(COALESCE(finished_at, reported_at), 1, 7)",
    "status": "status",
}
_WO_COLUMNS = (
    "wo_id, equipment_id, description, component, primary_code, secondary_code, work_type, "
    "priority, status, reported_at, finished_at, duration_h, labor_h"
)
_WO_COLUMNS_QUALIFIED = ", ".join(f"w.{c.strip()}" for c in _WO_COLUMNS.split(","))


def _wo_filter(
    store: PlantStore,
    equipment: str | None,
    start: str | None,
    end: str | None,
    work_type: str | None,
    component: str | None,
    primary_code: str | None,
    status: str | None,
) -> tuple[str, list[Any]]:
    clauses, params = [], []
    if equipment:
        clauses.append("equipment_id = ?")
        params.append(store.equipment_id(equipment))
    when = "COALESCE(finished_at, reported_at)"
    if (lo := parse_bound(start, end=False)) is not None:
        clauses.append(f"{when} >= ?")
        params.append(lo)
    if (hi := parse_bound(end, end=True)) is not None:
        clauses.append(f"{when} < ?")
        params.append(hi)
    if work_type:
        clauses.append("work_type = ?")
        params.append(work_type)
    if component:
        clauses.append("component LIKE ?")
        params.append(f"%{component}%")
    if primary_code:
        clauses.append("primary_code = ?")
        params.append(primary_code.upper())
    if status:
        clauses.append("status = ?")
        params.append(status.upper())
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def create_server(db: Path | PlantStore, *, read_only: bool = False) -> MCPServer:
    store = db if isinstance(db, PlantStore) else PlantStore(db)
    server = MCPServer(
        "maintenance",
        instructions=(
            "Work orders (PM = preventive, CM = corrective), events and alerts of the plant's "
            "equipment. Dates are ISO (YYYY, YYYY-MM or YYYY-MM-DD); end dates are inclusive."
        ),
        log_level="WARNING",
    )

    @server.tool(annotations=READ)
    def search_work_orders(
        equipment: str | None = None,
        start: str | None = None,
        end: str | None = None,
        work_type: WorkType | None = None,
        component: str | None = None,
        primary_code: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Find work orders by equipment, date range, type, component, failure code or status.
        Returns the total match count and up to `limit` work orders, newest first."""
        where, params = _wo_filter(
            store, equipment, start, end, work_type, component, primary_code, status
        )
        total = store.query(f"SELECT COUNT(*) AS n FROM work_orders {where}", params)[0]["n"]
        rows = store.query(
            f"SELECT {_WO_COLUMNS} FROM work_orders {where} "
            "ORDER BY COALESCE(finished_at, reported_at) DESC, wo_id LIMIT ?",
            [*params, max(1, min(limit, MAX_ROWS))],
        )
        return {"total": total, "returned": len(rows), "work_orders": rows}

    @server.tool(annotations=READ)
    def count_work_orders(
        group_by: GroupBy,
        equipment: str | None = None,
        start: str | None = None,
        end: str | None = None,
        work_type: WorkType | None = None,
        component: str | None = None,
    ) -> dict[str, Any]:
        """Count work orders grouped by work_type, component, primary_code, year, month or status,
        with the same filters as search_work_orders."""
        where, params = _wo_filter(store, equipment, start, end, work_type, component, None, None)
        key = _GROUP_SQL[group_by]
        rows = store.query(
            f"SELECT {key} AS k, COUNT(*) AS n FROM work_orders {where} GROUP BY k ORDER BY k",
            params,
        )
        return {
            "group_by": group_by,
            "total": sum(r["n"] for r in rows),
            "counts": {r["k"]: r["n"] for r in rows},
        }

    @server.tool(annotations=READ)
    def get_work_order(wo_id: str) -> dict[str, Any]:
        """One work order with its failure-code descriptions."""
        rows = store.query(
            f"SELECT {_WO_COLUMNS_QUALIFIED}, f.primary_description, f.secondary_description "
            "FROM work_orders w LEFT JOIN failure_codes f ON f.secondary_code = w.secondary_code "
            "WHERE wo_id = ?",
            [wo_id.strip().upper()],
        )
        if not rows:
            raise ToolError(f"no work order '{wo_id}'")
        return rows[0]

    @server.tool(annotations=READ)
    def list_events(
        equipment: str,
        start: str | None = None,
        end: str | None = None,
        event_group: Literal["WORK_ORDER", "ALERT", "ANOMALY"] | None = None,
        summarize_by: Literal["none", "group", "day"] = "group",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Events (work-order events, alerts, anomalies) of one equipment in a date range.
        summarize_by='group' counts per event group, 'day' counts per day and group, 'none'
        lists the events (up to `limit`, oldest first)."""
        clauses, params = ["equipment_id = ?"], [store.equipment_id(equipment)]
        if (lo := parse_bound(start, end=False)) is not None:
            clauses.append("event_time >= ?")
            params.append(lo)
        if (hi := parse_bound(end, end=True)) is not None:
            clauses.append("event_time < ?")
            params.append(hi)
        if event_group:
            clauses.append("event_group = ?")
            params.append(event_group)
        where = "WHERE " + " AND ".join(clauses)
        total = store.query(f"SELECT COUNT(*) AS n FROM events {where}", params)[0]["n"]
        if summarize_by == "group":
            rows = store.query(
                f"SELECT event_group AS k, COUNT(*) AS n FROM events {where} GROUP BY k", params
            )
            return {"total": total, "by_group": {r["k"]: r["n"] for r in rows}}
        if summarize_by == "day":
            rows = store.query(
                f"SELECT substr(event_time, 1, 10) AS d, event_group AS g, COUNT(*) AS n "
                f"FROM events {where} GROUP BY d, g ORDER BY d",
                params,
            )
            days: dict[str, dict[str, int]] = {}
            for r in rows:
                days.setdefault(r["d"], {})[r["g"]] = r["n"]
            return {"total": total, "by_day": days}
        rows = store.query(
            f"SELECT event_id, event_group, event_category, event_type, description, event_time, "
            f"note FROM events {where} ORDER BY event_time LIMIT ?",
            [*params, max(1, min(limit, MAX_ROWS))],
        )
        return {"total": total, "returned": len(rows), "events": rows}

    @server.tool(annotations=READ)
    def list_alerts(
        equipment: str | None = None,
        start: str | None = None,
        end: str | None = None,
        rule_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Alert occurrences (rule, start and end time), with counts per rule."""
        clauses, params = [], []
        if equipment:
            clauses.append("a.equipment_id = ?")
            params.append(store.equipment_id(equipment))
        if (lo := parse_bound(start, end=False)) is not None:
            clauses.append("a.start_time >= ?")
            params.append(lo)
        if (hi := parse_bound(end, end=True)) is not None:
            clauses.append("a.start_time < ?")
            params.append(hi)
        if rule_id:
            clauses.append("a.rule_id = ?")
            params.append(rule_id.upper())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        per_rule = store.query(
            f"SELECT a.rule_id, r.name, COUNT(*) AS n FROM alerts a "
            f"JOIN alert_rules r USING (rule_id) {where} GROUP BY a.rule_id ORDER BY n DESC",
            params,
        )
        rows = store.query(
            f"SELECT a.equipment_id, a.rule_id, a.start_time, a.end_time FROM alerts a {where} "
            "ORDER BY a.start_time LIMIT ?",
            [*params, max(1, min(limit, MAX_ROWS))],
        )
        return {
            "total": sum(r["n"] for r in per_rule),
            "by_rule": [
                {"rule_id": r["rule_id"], "name": r["name"], "count": r["n"]} for r in per_rule
            ],
            "alerts": rows,
        }

    if read_only:
        return server

    @server.tool(annotations=WRITE)
    def create_work_order(
        equipment: str,
        description: str,
        work_type: WorkType,
        priority: int,
        component: str = "entire chiller system",
        primary_code: str | None = None,
    ) -> dict[str, Any]:
        """Create a work order (status WAPPR, waiting for approval). priority: 1 = highest, 5 =
        lowest. primary_code: optional failure code such as 'M006'."""
        eq_id = store.equipment_id(equipment)
        if not 1 <= priority <= 5:
            raise ToolError("priority must be between 1 (highest) and 5 (lowest)")
        if not description.strip():
            raise ToolError("description must not be empty")
        if primary_code and not store.query(
            "SELECT 1 FROM failure_codes WHERE primary_code = ? LIMIT 1", [primary_code.upper()]
        ):
            raise ToolError(f"unknown failure code '{primary_code}'")
        # Numeric ids continue after the highest existing one (ids differ in length, so compare
        # numbers, not strings).
        last = store.query("SELECT MAX(CAST(substr(wo_id, 3) AS INTEGER)) AS m FROM work_orders")
        wo_id = f"WO{int(last[0]['m'] or 0) + 1}"
        store.execute(
            "INSERT INTO work_orders (wo_id, equipment_id, description, component, primary_code, "
            "work_type, priority, status, reported_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'WAPPR', ?)",
            [
                wo_id,
                eq_id,
                description.strip()[:200],
                component,
                primary_code and primary_code.upper(),
                work_type,
                priority,
                _now(),
            ],
        )
        return {"created": wo_id, "status": "WAPPR"}

    @server.tool(annotations=WRITE)
    def update_work_order(
        wo_id: str, priority: int | None = None, status: str | None = None
    ) -> dict[str, Any]:
        """Change an open work order's priority (1-5) or status (WAPPR, APPR, INPRG)."""
        wo = _open_work_order(store, wo_id)
        if priority is not None and not 1 <= priority <= 5:
            raise ToolError("priority must be between 1 (highest) and 5 (lowest)")
        if status is not None and status.upper() not in OPEN_STATUSES:
            raise ToolError(
                f"status must be one of {', '.join(OPEN_STATUSES)}; "
                "use close_work_order or cancel_work_order to finish one"
            )
        store.execute(
            "UPDATE work_orders SET priority = COALESCE(?, priority), "
            "status = COALESCE(?, status) WHERE wo_id = ?",
            [priority, status and status.upper(), wo["wo_id"]],
        )
        return {"updated": wo["wo_id"]}

    @server.tool(annotations=DESTRUCTIVE)
    def close_work_order(wo_id: str) -> dict[str, Any]:
        """Close an open work order as completed."""
        wo = _open_work_order(store, wo_id)
        store.execute(
            "UPDATE work_orders SET status = 'CLOSE', finished_at = ? WHERE wo_id = ?",
            [_now(), wo["wo_id"]],
        )
        return {"closed": wo["wo_id"]}

    @server.tool(annotations=DESTRUCTIVE)
    def cancel_work_order(wo_id: str, reason: str) -> dict[str, Any]:
        """Cancel an open work order, giving a reason."""
        wo = _open_work_order(store, wo_id)
        if not reason.strip():
            raise ToolError("a reason is required to cancel a work order")
        store.execute("UPDATE work_orders SET status = 'CAN' WHERE wo_id = ?", [wo["wo_id"]])
        return {"cancelled": wo["wo_id"]}

    return server


def _open_work_order(store: PlantStore, wo_id: str) -> dict[str, Any]:
    rows = store.query("SELECT * FROM work_orders WHERE wo_id = ?", [wo_id.strip().upper()])
    if not rows:
        raise ToolError(f"no work order '{wo_id}'")
    if rows[0]["status"] not in OPEN_STATUSES:
        raise ToolError(f"work order {rows[0]['wo_id']} is {rows[0]['status']}, not open")
    return rows[0]


def _now() -> str:
    """Fixed clock: the plant data ends in October 2023, so 'now' is the day after."""
    return "2023-10-13T08:00:00"
