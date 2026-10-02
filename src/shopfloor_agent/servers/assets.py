"""Assets MCP server: the equipment registry, components and installed sensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from shopfloor_agent.servers.maintenance import READ
from shopfloor_agent.servers.store import PlantStore


def create_server(db: Path | PlantStore) -> MCPServer:
    store = db if isinstance(db, PlantStore) else PlantStore(db)
    server = MCPServer(
        "assets",
        instructions="Equipment registry: ids, names, asset classes, components and sensors.",
        log_level="WARNING",
    )

    @server.tool(annotations=READ)
    def list_equipment(asset_class: str | None = None) -> dict[str, Any]:
        """All equipment (id, name, asset class), optionally of one asset class ('chiller')."""
        rows = [
            e
            for e in store.all_equipment()
            if asset_class is None or e["asset_class"] == asset_class.strip().lower()
        ]
        return {"count": len(rows), "equipment": rows}

    @server.tool(annotations=READ)
    def get_equipment(equipment: str) -> dict[str, Any]:
        """One equipment by id or name, with its sensors and its first and last work order."""
        eq = store.equipment(equipment)
        sensors = store.query(
            "SELECT sensor_id, name FROM sensors WHERE equipment_id = ? ORDER BY sensor_id",
            [eq["equipment_id"]],
        )
        span = store.query(
            "SELECT MIN(finished_at) AS first, MAX(finished_at) AS last, COUNT(*) AS n "
            "FROM work_orders WHERE equipment_id = ?",
            [eq["equipment_id"]],
        )[0]
        return {**eq, "sensors": sensors, "work_orders": span}

    @server.tool(annotations=READ)
    def list_components(asset_class: str = "chiller") -> dict[str, Any]:
        """The components of an asset class and what each one does."""
        rows = store.query(
            "SELECT component, explanation FROM components WHERE asset_class = ? "
            "ORDER BY component",
            [asset_class.strip().lower()],
        )
        return {"asset_class": asset_class, "components": rows}

    return server
