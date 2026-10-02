"""Telemetry MCP server: sensor readings (15-minute intervals) and statistics over them."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from shopfloor_agent.servers.maintenance import READ
from shopfloor_agent.servers.store import PlantStore, ToolError, parse_bound

MAX_POINTS = 96  # one day at 15-minute resolution


def create_server(db: Path | PlantStore) -> MCPServer:
    store = db if isinstance(db, PlantStore) else PlantStore(db)
    server = MCPServer(
        "telemetry",
        instructions=(
            "Sensor readings of the plant's equipment. Use sensor_stats for summaries over long "
            "ranges; sensor_history returns at most 96 readings."
        ),
        log_level="WARNING",
    )

    def resolve_sensor(name: str, equipment: str | None) -> dict[str, Any]:
        """A sensor id ('CWC04006.power_input'), or a sensor name together with its equipment."""
        rows = store.query("SELECT * FROM sensors WHERE sensor_id = ?", [name.strip()])
        if not rows and equipment:
            rows = store.query(
                "SELECT * FROM sensors WHERE equipment_id = ? AND lower(name) = ?",
                [store.equipment_id(equipment), name.strip().lower()],
            )
        if not rows:
            raise ToolError(f"unknown sensor '{name}'; call list_sensors for valid sensor ids")
        return rows[0]

    def window(sensor_id: str, start: str | None, end: str | None) -> tuple[str, list[Any]]:
        clauses, params = ["sensor_id = ?"], [sensor_id]
        if (lo := parse_bound(start, end=False)) is not None:
            clauses.append("ts >= ?")
            params.append(lo)
        if (hi := parse_bound(end, end=True)) is not None:
            clauses.append("ts < ?")
            params.append(hi)
        return "WHERE " + " AND ".join(clauses), params

    @server.tool(annotations=READ)
    def list_sensors(equipment: str) -> dict[str, Any]:
        """Sensors installed on one equipment, with the time range of their readings."""
        eq_id = store.equipment_id(equipment)
        rows = store.query(
            "SELECT s.sensor_id, s.name, COUNT(t.ts) AS readings, MIN(t.ts) AS first, "
            "MAX(t.ts) AS last FROM sensors s LEFT JOIN telemetry t USING (sensor_id) "
            "WHERE s.equipment_id = ? GROUP BY s.sensor_id ORDER BY s.sensor_id",
            [eq_id],
        )
        return {"equipment_id": eq_id, "sensors": rows}

    @server.tool(annotations=READ)
    def sensor_stats(
        sensor: str,
        start: str | None = None,
        end: str | None = None,
        equipment: str | None = None,
    ) -> dict[str, Any]:
        """Count, mean, min, max and standard deviation of one sensor over a date range, with the
        times of the minimum and maximum."""
        s = resolve_sensor(sensor, equipment)
        where, params = window(s["sensor_id"], start, end)
        agg = store.query(
            f"SELECT COUNT(*) AS n, AVG(value) AS mean, MIN(value) AS min, MAX(value) AS max, "
            f"AVG(value * value) AS mean_sq FROM telemetry {where}",
            params,
        )[0]
        if not agg["n"]:
            return {"sensor_id": s["sensor_id"], "count": 0}
        at_min = store.query(f"SELECT ts FROM telemetry {where} ORDER BY value, ts LIMIT 1", params)
        at_max = store.query(
            f"SELECT ts FROM telemetry {where} ORDER BY value DESC, ts LIMIT 1", params
        )
        std = math.sqrt(max(0.0, agg["mean_sq"] - agg["mean"] ** 2))
        return {
            "sensor_id": s["sensor_id"],
            "name": s["name"],
            "count": agg["n"],
            "mean": round(agg["mean"], 4),
            "min": round(agg["min"], 4),
            "min_at": at_min[0]["ts"],
            "max": round(agg["max"], 4),
            "max_at": at_max[0]["ts"],
            "std": round(std, 4),
        }

    @server.tool(annotations=READ)
    def sensor_history(
        sensor: str,
        start: str | None = None,
        end: str | None = None,
        equipment: str | None = None,
    ) -> dict[str, Any]:
        """Readings of one sensor in a date range, oldest first (at most 96; narrow the range or
        use sensor_stats for longer periods)."""
        s = resolve_sensor(sensor, equipment)
        where, params = window(s["sensor_id"], start, end)
        total = store.query(f"SELECT COUNT(*) AS n FROM telemetry {where}", params)[0]["n"]
        rows = store.query(
            f"SELECT ts, round(value, 4) AS value FROM telemetry {where} ORDER BY ts LIMIT ?",
            [*params, MAX_POINTS],
        )
        return {
            "sensor_id": s["sensor_id"],
            "total": total,
            "returned": len(rows),
            "readings": rows,
        }

    @server.tool(annotations=READ)
    def latest_reading(sensor: str, equipment: str | None = None) -> dict[str, Any]:
        """The most recent reading of one sensor."""
        s = resolve_sensor(sensor, equipment)
        rows = store.query(
            "SELECT ts, round(value, 4) AS value FROM telemetry WHERE sensor_id = ? "
            "ORDER BY ts DESC LIMIT 1",
            [s["sensor_id"]],
        )
        return {"sensor_id": s["sensor_id"], **(rows[0] if rows else {})}

    return server
