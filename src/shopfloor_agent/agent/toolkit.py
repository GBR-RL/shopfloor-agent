"""The agent's view of the plant: MCP sessions to the four servers, exposed as LangChain tools.

Every tool call goes through `Toolkit.call`, which records it (name, arguments, result, error,
latency). That record is what the evaluation scores and where the security layer hooks in.
The adapter is small on purpose: tool schemas are compacted for small models, and the result
text the model sees is produced in one place.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp.client import Client
from mcp.server.mcpserver import MCPServer

from shopfloor_agent.servers import assets, maintenance, reliability, telemetry


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    seconds: float = 0.0
    read_only: bool = True


def plant_servers(db_path: Path, *, read_only: bool = False) -> list[MCPServer]:
    return [
        assets.create_server(db_path),
        telemetry.create_server(db_path),
        maintenance.create_server(db_path, read_only=read_only),
        reliability.create_server(db_path),
    ]


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drops what small models do not need: titles, `null` alternatives of optional
    parameters, and `None` defaults (an omitted argument already means None)."""
    props = {}
    for name, prop in schema.get("properties", {}).items():
        p = {k: v for k, v in prop.items() if k != "title"}
        if "anyOf" in p:
            options = [o for o in p.pop("anyOf") if o.get("type") != "null"]
            if len(options) == 1:
                p = {**options[0], **p}
            else:
                p["anyOf"] = options
        if p.get("default", 0) is None:
            p.pop("default")
        props[name] = p
    out: dict[str, Any] = {"type": "object", "properties": props}
    if schema.get("required"):
        out["required"] = schema["required"]
    return out


ResultHook = Callable[[ToolCall], str]


def default_render(call: ToolCall) -> str:
    if call.error is not None:
        return f"ERROR: {call.error}"
    return json.dumps(call.result, ensure_ascii=False, default=str)


@dataclass
class Toolkit:
    """Open with `async with Toolkit(servers) as kit:`; `kit.tools` are the LangChain tools."""

    servers: Sequence[MCPServer]
    render: ResultHook = default_render
    calls: list[ToolCall] = field(default_factory=list)
    tools: list[StructuredTool] = field(default_factory=list)
    _clients: dict[str, Client] = field(default_factory=dict)
    _read_only: dict[str, bool] = field(default_factory=dict)
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def __aenter__(self) -> Toolkit:
        for server in self.servers:
            client = await self._stack.enter_async_context(Client(server))
            for tool in (await client.list_tools()).tools:
                if tool.name in self._clients:
                    raise ValueError(f"tool '{tool.name}' is defined by two servers")
                self._clients[tool.name] = client
                hints = tool.annotations
                self._read_only[tool.name] = bool(hints and hints.read_only_hint)
                self.tools.append(self._wrap(tool.name, tool.description or "", tool.input_schema))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    def is_read_only(self, name: str) -> bool:
        return self._read_only.get(name, False)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        record = ToolCall(name, dict(arguments), read_only=self.is_read_only(name))
        start = time.perf_counter()
        client = self._clients.get(name)
        if client is None:
            record.error = f"unknown tool '{name}'"
        else:
            result = await client.call_tool(name, arguments)
            text = " ".join(getattr(c, "text", "") for c in result.content)
            if result.is_error:
                record.error = text.removeprefix(f"Error executing tool {name}: ")
            else:
                record.result = result.structured_content or text
        record.seconds = time.perf_counter() - start
        self.calls.append(record)
        return record

    def _wrap(self, name: str, description: str, schema: dict[str, Any]) -> StructuredTool:
        async def run(**kwargs: Any) -> str:
            return self.render(await self.call(name, kwargs))

        return StructuredTool(
            name=name,
            description=" ".join(description.split()),
            args_schema=compact_schema(schema),
            coroutine=run,
        )
