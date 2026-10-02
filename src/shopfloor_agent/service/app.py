"""HTTP service: ask the agent a question and watch it work, approving every change it wants
to make.

POST /ask streams the run as server-sent events (each tool call as it starts and ends, a pending
approval when the agent wants to write, then the answer). A write waits until someone answers
POST /approvals/{id}; without a decision it is refused after `approval_timeout_s`. In
`writes="off"` mode the session has no write tools at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import generate_latest
from pydantic import BaseModel, Field

from shopfloor_agent.agent.graphs import DESIGNS, SYSTEM_PROMPT, Run, make_llm
from shopfloor_agent.agent.toolkit import ToolCall, Toolkit, default_render, plant_servers
from shopfloor_agent.config import Settings, get_settings
from shopfloor_agent.eval.check import extract_answer
from shopfloor_agent.servers.store import PlantStore
from shopfloor_agent.service.telemetry import Metrics, ModelSpans, setup_tracing, tracer

STATIC = Path(__file__).parent / "static"
AgentFactory = Callable[[str, Toolkit, str], Awaitable[Run]]  # (question, kit, design) -> Run


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    agent: str = "react"
    writes: Literal["approve", "off"] = "approve"


class Decision(BaseModel):
    approve: bool
    note: str = ""


def sse(kind: str, **data: Any) -> str:
    return f"data: {json.dumps({'type': kind, **data}, default=str)}\n\n"


@dataclass
class AskSession:
    """One question: runs the agent and turns what it does into a stream of events."""

    body: AskRequest
    store: PlantStore
    metrics: Metrics
    pending: dict[str, asyncio.Future[Decision]]
    run_agent: AgentFactory
    model: str
    approval_timeout_s: float
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    spans: dict[int, Any] = field(default_factory=dict)

    async def on_event(self, phase: str, call: ToolCall) -> None:
        if phase == "start":
            span = tracer.start_span(f"execute_tool {call.name}")
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", call.name)
            self.spans[id(call)] = span
            await self.queue.put(sse("tool_start", tool=call.name, args=call.arguments))
            return
        outcome = "blocked" if call.blocked else ("error" if call.error else "ok")
        self.metrics.tool_calls.labels(call.name, outcome).inc()
        self.metrics.tool_seconds.labels(call.name).observe(call.seconds)
        if span := self.spans.pop(id(call), None):
            span.set_attribute("shopfloor.outcome", outcome)
            span.end()
        result = None if call.error else default_render(call)[:400]
        await self.queue.put(sse("tool_end", tool=call.name, outcome=outcome, error=call.error,
                                 seconds=round(call.seconds, 3), result=result))  # fmt: skip

    async def approve(self, call: ToolCall) -> str | None:
        if call.read_only:
            return None
        approval_id = uuid.uuid4().hex[:12]
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self.pending[approval_id] = future
        await self.queue.put(sse("approval_required", id=approval_id, tool=call.name,
                                 args=call.arguments))  # fmt: skip
        try:
            decision = await asyncio.wait_for(future, self.approval_timeout_s)
        except TimeoutError:
            self.metrics.approvals.labels("timeout").inc()
            return f"no decision within {self.approval_timeout_s:.0f} s"
        finally:
            self.pending.pop(approval_id, None)
        self.metrics.approvals.labels("approved" if decision.approve else "rejected").inc()
        await self.queue.put(sse("approval_decided", id=approval_id, approved=decision.approve))
        return None if decision.approve else f"rejected by the operator {decision.note}".strip()

    async def work(self) -> None:
        start, outcome = time.perf_counter(), "error"
        with tracer.start_as_current_span("invoke_agent") as span:
            span.set_attribute("gen_ai.operation.name", "invoke_agent")
            span.set_attribute("gen_ai.agent.name", self.body.agent)
            span.set_attribute("gen_ai.request.model", self.model)
            try:
                servers = plant_servers(self.store, read_only=self.body.writes == "off")
                async with Toolkit(servers, approve=self.approve, on_event=self.on_event) as kit:
                    run = await self.run_agent(self.body.question, kit, self.body.agent)
                outcome = "answered" if run.answer else run.stopped
                await self.queue.put(sse("answer", text=run.answer,
                                         answer=extract_answer(run.answer), steps=run.steps,
                                         seconds=round(run.seconds, 2), tokens_in=run.tokens_in,
                                         tokens_out=run.tokens_out))  # fmt: skip
            except Exception as exc:  # reported to the client; the service keeps running
                span.record_exception(exc)
                await self.queue.put(sse("error", message=f"{type(exc).__name__}: {exc}"))
            finally:
                span.set_attribute("shopfloor.outcome", outcome)
                self.metrics.requests.labels(self.body.agent, outcome).inc()
                self.metrics.request_seconds.labels(self.body.agent).observe(
                    time.perf_counter() - start
                )
                await self.queue.put(None)

    async def stream(self) -> AsyncIterator[str]:
        task = asyncio.create_task(self.work())
        try:
            yield sse("started", question=self.body.question, agent=self.body.agent,
                      writes=self.body.writes)  # fmt: skip
            while (item := await self.queue.get()) is not None:
                yield item
        finally:
            if not task.done():
                task.cancel()


def create_app(
    settings: Settings | None = None,
    *,
    agent_factory: AgentFactory | None = None,
    approval_timeout_s: float = 600.0,
) -> FastAPI:
    settings = settings or get_settings()
    metrics = Metrics()
    pending: dict[str, asyncio.Future[Decision]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_tracing()
        app.state.store = PlantStore(settings.plant_db)
        yield
        app.state.store.close()

    app = FastAPI(title="shopfloor-agent", lifespan=lifespan)

    async def default_agent(question: str, kit: Toolkit, design: str) -> Run:
        llm = make_llm(settings, callbacks=[ModelSpans(metrics, settings.llm_model)])
        return await DESIGNS[design](question, kit, llm, system=SYSTEM_PROMPT)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        reachable = False
        with contextlib.suppress(httpx.HTTPError):
            async with httpx.AsyncClient(timeout=2) as client:
                reachable = (await client.get(f"{settings.llm_base_url}/models")).is_success
        return {"status": "ok", "model": settings.llm_model, "llm_reachable": reachable}

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type="text/plain; version=0.0.4")

    @app.get("/")
    async def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/approvals/{approval_id}")
    async def decide(approval_id: str, decision: Decision) -> dict[str, Any]:
        future = pending.get(approval_id)
        if future is None or future.done():
            raise HTTPException(404, f"no pending approval '{approval_id}'")
        future.set_result(decision)
        return {"id": approval_id, "approved": decision.approve}

    @app.post("/ask")
    async def ask(body: AskRequest, request: Request) -> StreamingResponse:
        if body.agent not in DESIGNS:
            raise HTTPException(422, f"agent must be one of {', '.join(DESIGNS)}")
        session = AskSession(body, request.app.state.store, metrics, pending,
                             agent_factory or default_agent, settings.llm_model,
                             approval_timeout_s)  # fmt: skip
        return StreamingResponse(session.stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})  # fmt: skip

    return app
