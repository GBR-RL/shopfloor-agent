from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from shopfloor_agent.agent.toolkit import Toolkit, compact_schema, plant_servers

pytestmark = pytest.mark.anyio


def test_compact_schema_drops_null_alternatives_and_titles() -> None:
    schema = {
        "type": "object",
        "title": "args",
        "properties": {
            "equipment": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Equipment",
            },
            "work_type": {
                "anyOf": [{"enum": ["PM", "CM"], "type": "string"}, {"type": "null"}],
                "default": None,
            },
            "limit": {"type": "integer", "default": 20, "title": "Limit"},
        },
        "required": [],
    }
    assert compact_schema(schema) == {
        "type": "object",
        "properties": {
            "equipment": {"type": "string"},
            "work_type": {"enum": ["PM", "CM"], "type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
    }


async def test_toolkit_records_calls_and_exposes_tiers(plant_db: Path) -> None:
    async with Toolkit(plant_servers(plant_db)) as kit:
        names = [t.name for t in kit.tools]
        assert len(names) == len(set(names)) > 15
        assert kit.is_read_only("search_work_orders")
        assert not kit.is_read_only("close_work_order")
        tool = next(t for t in kit.tools if t.name == "count_work_orders")
        text = await tool.ainvoke({"group_by": "year", "equipment": "Chiller 6"})
        assert '"group_by": "year"' in text
        await kit.call("get_work_order", {"wo_id": "nope"})
    first, second = kit.calls
    assert first.name == "count_work_orders"
    assert first.error is None
    assert first.result["total"] > 0
    assert second.error == "no work order 'nope'"


async def test_duplicate_tool_names_are_rejected(plant_db: Path) -> None:
    def server(name: str) -> MCPServer:
        s = MCPServer(name)

        @s.tool()
        def ping() -> str:
            return "pong"

        return s

    with pytest.raises(ValueError, match="defined by two servers"):
        async with Toolkit([server("a"), server("b")]):
            pass
