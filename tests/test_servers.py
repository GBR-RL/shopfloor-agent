"""The MCP servers, exercised through real MCP client sessions (in-memory transport)."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from mcp.server.mcpserver import MCPServer

from shopfloor_agent.servers import assets, maintenance, reliability, telemetry

pytestmark = pytest.mark.anyio


async def call(server: MCPServer, tool: str, **args: Any) -> tuple[bool, Any]:
    async with Client(server) as client:
        result = await client.call_tool(tool, args)
        if result.is_error:
            return True, result.content[0].text
        return False, result.structured_content


def sql(db: Path, query: str, *params: Any) -> Any:
    return sqlite3.connect(db).execute(query, params).fetchone()[0]


async def test_counts_match_the_database(plant_db: Path) -> None:
    server = maintenance.create_server(plant_db)
    err, out = await call(
        server,
        "count_work_orders",
        group_by="work_type",
        equipment="Chiller 9",
        start="2017",
        end="2017",
    )
    assert not err
    expected_cm = sql(
        plant_db,
        "SELECT COUNT(*) FROM work_orders WHERE equipment_id = 'CWC04009' "
        "AND work_type = 'CM' AND finished_at LIKE '2017%'",
    )
    assert out["counts"]["CM"] == expected_cm
    # the same count through search, by id instead of name, and with day-precise bounds
    err, out = await call(
        server,
        "search_work_orders",
        equipment="CWC04009",
        work_type="CM",
        start="2017-01-01",
        end="2017-12-31",
        limit=5,
    )
    assert out["total"] == expected_cm
    assert out["returned"] == 5


async def test_errors_reach_the_agent_with_guidance(plant_db: Path) -> None:
    server = maintenance.create_server(plant_db)
    err, text = await call(server, "search_work_orders", equipment="Chiller 99")
    assert err
    assert "unknown equipment 'Chiller 99'" in text
    assert "Chiller 6" in text  # lists the valid names
    err, text = await call(server, "list_events", equipment="Chiller 6", start="yesterday")
    assert err
    assert "invalid date" in text


async def test_write_tier_is_absent_in_read_only_mode(plant_db: Path) -> None:
    async with Client(maintenance.create_server(plant_db, read_only=True)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert "search_work_orders" in names
    assert not names & {
        "create_work_order",
        "update_work_order",
        "close_work_order",
        "cancel_work_order",
    }
    async with Client(maintenance.create_server(plant_db)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert tools["close_work_order"].annotations.destructive_hint is True
    assert tools["search_work_orders"].annotations.read_only_hint is True


async def test_work_order_lifecycle(plant_db: Path) -> None:
    server = maintenance.create_server(plant_db)
    highest = sql(plant_db, "SELECT MAX(CAST(substr(wo_id, 3) AS INTEGER)) FROM work_orders")
    err, out = await call(
        server,
        "create_work_order",
        equipment="chiller6",
        description="Inspect condenser water flow",
        work_type="CM",
        priority=2,
        primary_code="m015",
    )
    assert not err
    wo_id = out["created"]
    assert wo_id == f"WO{highest + 1}"
    assert sql(plant_db, "SELECT status FROM work_orders WHERE wo_id = ?", wo_id) == "WAPPR"
    assert sql(plant_db, "SELECT primary_code FROM work_orders WHERE wo_id = ?", wo_id) == "M015"
    assert (await call(server, "update_work_order", wo_id=wo_id, priority=1))[0] is False
    assert (await call(server, "close_work_order", wo_id=wo_id))[0] is False
    assert sql(plant_db, "SELECT status FROM work_orders WHERE wo_id = ?", wo_id) == "CLOSE"
    err, text = await call(server, "cancel_work_order", wo_id=wo_id, reason="duplicate")
    assert err
    assert "not open" in text  # a closed order cannot be cancelled
    err, text = await call(
        server,
        "create_work_order",
        equipment="Chiller 6",
        description="x",
        work_type="CM",
        priority=9,
    )
    assert err
    assert "priority" in text


async def test_events_and_alerts(plant_db: Path) -> None:
    server = maintenance.create_server(plant_db)
    err, out = await call(
        server, "list_events", equipment="Chiller 9", start="2020-06", end="2020-06"
    )
    assert not err
    expected = sql(
        plant_db,
        "SELECT COUNT(*) FROM events WHERE equipment_id = 'CWC04009' "
        "AND event_time LIKE '2020-06%'",
    )
    assert out["total"] == expected == sum(out["by_group"].values())
    err, out = await call(server, "list_alerts", equipment="Chiller 6", limit=3)
    assert out["total"] == sql(
        plant_db, "SELECT COUNT(*) FROM alerts WHERE equipment_id = 'CWC04006'"
    )
    assert len(out["alerts"]) == 3


async def test_telemetry_stats(plant_db: Path) -> None:
    server = telemetry.create_server(plant_db)
    err, out = await call(
        server,
        "sensor_stats",
        sensor="Power Input",
        equipment="Chiller 6",
        start="2020-06-07",
        end="2020-06-07",
    )
    assert not err
    assert out["count"] == 96  # one day at 15-minute resolution
    expected_max = sql(
        plant_db,
        "SELECT MAX(value) FROM telemetry WHERE sensor_id = "
        "'CWC04006.power_input' AND ts LIKE '2020-06-07%'",
    )
    assert out["max"] == pytest.approx(expected_max, abs=1e-3)
    err, out = await call(server, "sensor_history", sensor="CWC04006.power_input")
    assert out["returned"] == telemetry.MAX_POINTS < out["total"]
    err, text = await call(server, "latest_reading", sensor="flux capacitor")
    assert err
    assert "list_sensors" in text


async def test_assets_and_reliability(plant_db: Path) -> None:
    err, out = await call(assets.create_server(plant_db), "list_equipment")
    assert out["count"] == 11
    assert [e["name"] for e in out["equipment"]][:3] == ["Chiller 1", "Chiller 2", "Chiller 3"]
    server = reliability.create_server(plant_db)
    err, out = await call(server, "failure_codes_for_alert", rule_id="rul0018")
    assert not err
    assert out["failure_codes"]
    err, out = await call(server, "find_failure_codes", query="overheating")
    assert all(
        "overheat" in (r["primary_description"] + r["secondary_description"]).lower()
        for r in out["failure_codes"]
    )
    err, text = await call(server, "list_failure_modes", asset_class="boiler")
    assert err
    assert "chiller" in text
