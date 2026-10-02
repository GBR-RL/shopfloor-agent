"""The HTTP service over a real socket (uvicorn in a thread), with scripted agents: the event
stream, the approval round trip, and the read-only mode."""

import json
import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from shopfloor_agent.agent.graphs import Run
from shopfloor_agent.agent.toolkit import Toolkit
from shopfloor_agent.config import Settings
from shopfloor_agent.service.app import create_app


async def scripted(question: str, kit: Toolkit, design: str) -> Run:
    """Counts work orders; if asked to, also files one (a write that needs approval)."""
    found = await kit.call("count_work_orders", {"group_by": "work_type", "equipment": "Chiller 9",
                                                 "start": "2017", "end": "2017"})  # fmt: skip
    if "file" in question:
        made = await kit.call("create_work_order", {"equipment": "Chiller 9", "work_type": "CM",
                                                    "priority": 2,
                                                    "description": "Check flow"})  # fmt: skip
        status = "created" if made.error is None else f"not created ({made.error})"
        return Run(answer=f"Work order {status}.\nANSWER: done", steps=2, seconds=0.0)
    return Run(answer=f"ANSWER: {found.result['counts']['CM']}", steps=1, seconds=0.0)


@pytest.fixture
def server(plant_db: Path) -> Iterator[tuple[str, Path]]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(Settings(data_dir=plant_db.parent, llm_base_url="http://127.0.0.1:1/v1"),
                     agent_factory=scripted, approval_timeout_s=20)  # fmt: skip
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{base}/health", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield base, plant_db
    srv.should_exit = True
    thread.join(timeout=10)


def ask(base: str, question: str, decide: bool | None = None, **body: Any) -> list[dict[str, Any]]:
    """Reads the event stream; answers an approval request with `decide` if one arrives."""
    events = []
    with httpx.stream("POST", f"{base}/ask", json={"question": question, **body},
                      timeout=60) as response:  # fmt: skip
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            events.append(event)
            if event["type"] == "approval_required" and decide is not None:
                r = httpx.post(f"{base}/approvals/{event['id']}", json={"approve": decide})
                assert r.status_code == 200
    return events


def test_stream_shows_each_step_and_the_answer(server: tuple[str, Path]) -> None:
    base, _ = server
    events = ask(base, "How many corrective work orders did Chiller 9 have in 2017?")
    kinds = [e["type"] for e in events]
    assert kinds == ["started", "tool_start", "tool_end", "answer"]
    assert events[2]["outcome"] == "ok"
    assert events[3]["answer"] == "20"


def test_write_waits_for_approval(server: tuple[str, Path]) -> None:
    base, db = server
    count = "SELECT COUNT(*) FROM work_orders WHERE status = 'WAPPR'"
    before = sqlite3.connect(db).execute(count).fetchone()[0]
    events = ask(base, "Please file a work order for Chiller 9", decide=True)
    kinds = [e["type"] for e in events]
    assert "approval_required" in kinds
    assert kinds.index("approval_required") < kinds.index("approval_decided")
    assert [e["outcome"] for e in events if e["type"] == "tool_end"] == ["ok", "ok"]
    assert sqlite3.connect(db).execute(count).fetchone()[0] == before + 1

    events = ask(base, "Please file a work order for Chiller 9", decide=False)
    assert [e["outcome"] for e in events if e["type"] == "tool_end"] == ["ok", "blocked"]
    assert "rejected by the operator" in events[-1]["text"]
    assert sqlite3.connect(db).execute(count).fetchone()[0] == before + 1  # unchanged


def test_writes_off_and_metrics(server: tuple[str, Path]) -> None:
    base, _ = server
    events = ask(base, "Please file a work order for Chiller 9", writes="off")
    assert "approval_required" not in [e["type"] for e in events]
    assert "unknown tool 'create_work_order'" in events[-1]["text"]
    metrics = httpx.get(f"{base}/metrics").text
    assert 'shopfloor_tool_calls_total{outcome="ok",tool="count_work_orders"}' in metrics
    assert 'shopfloor_requests_total{agent="react",outcome="answered"}' in metrics
    assert httpx.get(f"{base}/health").json()["llm_reachable"] is False
    assert httpx.post(f"{base}/approvals/nope", json={"approve": True}).status_code == 404
