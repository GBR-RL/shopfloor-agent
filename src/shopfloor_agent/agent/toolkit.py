"""The agent's view of the plant: MCP sessions to the four servers, exposed as LangChain tools.

Every tool call goes through `Toolkit.call`, which records it (name, arguments, result, error,
latency). That record is what the evaluation scores and where the security layer hooks in.
The adapter is small on purpose: tool schemas are compacted for small models, and the result
text the model sees is produced in one place.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp.client import Client
from mcp.server.mcpserver import MCPServer

from shopfloor_agent.servers import assets, maintenance, reliability, telemetry
from shopfloor_agent.servers.store import PlantStore


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    seconds: float = 0.0
    read_only: bool = True
    blocked: bool = False  # stopped by the approval gate before reaching the server


def plant_servers(db: Path | PlantStore, *, read_only: bool = False) -> list[MCPServer]:
    """The four plant servers, sharing one database connection."""
    store = db if isinstance(db, PlantStore) else PlantStore(db)
    return [
        assets.create_server(store),
        telemetry.create_server(store),
        maintenance.create_server(store, read_only=read_only),
        reliability.create_server(store),
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
# None approves; a string is the refusal. May be async (a person deciding in the service).
ApprovalHook = Callable[[ToolCall], "str | Awaitable[str | None] | None"]
EventHook = Callable[[str, ToolCall], "Awaitable[None] | None"]  # ("start" | "end", call)


def default_render(call: ToolCall) -> str:
    if call.error is not None:
        return f"ERROR: {call.error}"
    # compact separators: on a CPU every prompt token costs time
    return json.dumps(call.result, ensure_ascii=False, default=str, separators=(",", ":"))


@dataclass
class Toolkit:
    """Open with `async with Toolkit(servers) as kit:`; `kit.tools` are the LangChain tools."""

    servers: Sequence[MCPServer]
    render: ResultHook = default_render
    approve: ApprovalHook | None = None
    on_event: EventHook | None = None
    calls: list[ToolCall] = field(default_factory=list)
    tools: list[StructuredTool] = field(default_factory=list)
    instructions: dict[str, str] = field(default_factory=dict)  # per server, from MCP init
    _clients: dict[str, Client] = field(default_factory=dict)
    _read_only: dict[str, bool] = field(default_factory=dict)
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def __aenter__(self) -> Toolkit:
        for server in self.servers:
            client = await self._stack.enter_async_context(Client(server))
            if client.instructions:
                self.instructions[server.name or "server"] = " ".join(client.instructions.split())
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
        await self._emit("start", record)
        client = self._clients.get(name)
        refusal = None
        if self.approve is not None and client is not None:
            decision = self.approve(record)
            refusal = await decision if inspect.isawaitable(decision) else decision
        if client is None:
            record.error = f"unknown tool '{name}'"
        elif refusal is not None:
            record.error, record.blocked = f"not approved: {refusal}", True
        else:
            result = await client.call_tool(name, arguments)
            text = " ".join(getattr(c, "text", "") for c in result.content)
            if result.is_error:
                record.error = text.removeprefix(f"Error executing tool {name}: ")
            else:
                record.result = result.structured_content or text
        record.seconds = time.perf_counter() - start
        self.calls.append(record)
        await self._emit("end", record)
        return record

    async def _emit(self, phase: str, call: ToolCall) -> None:
        if self.on_event is not None:
            out = self.on_event(phase, call)
            if inspect.isawaitable(out):
                await out

    def _wrap(self, name: str, description: str, schema: dict[str, Any]) -> StructuredTool:
        async def run(**kwargs: Any) -> str:
            return self.render(await self.call(name, kwargs))

        return StructuredTool(
            name=name,
            description=" ".join(description.split()),
            args_schema=compact_schema(schema),
            coroutine=run,
        )
