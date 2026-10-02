"""Runs tasks as isolated episodes and writes one result row per task.

Each episode gets a fresh copy of the plant database with the task's setup rows, so write tools
cannot leak between tasks. Results are appended to a JSONL file as they finish; a rerun skips
the tasks already there, so an interrupted benchmark resumes.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import anyio

from shopfloor_agent.agent.graphs import DESIGNS, Run, make_llm
from shopfloor_agent.agent.toolkit import Toolkit, plant_servers
from shopfloor_agent.config import Settings
from shopfloor_agent.eval.check import score, snapshot
from shopfloor_agent.eval.tasks import Task
from shopfloor_agent.servers.store import PlantStore

AgentFn = Callable[[str, Toolkit], Awaitable[Run]]


def prepare_episode(base_db: Path, task: Task, workdir: Path) -> Path:
    db_path = workdir / "episode.db"
    shutil.copyfile(base_db, db_path)
    if task.setup:
        db = sqlite3.connect(db_path)
        try:
            for row in task.setup:
                db.execute(
                    f"INSERT INTO work_orders ({', '.join(row)}) VALUES "
                    f"({', '.join('?' * len(row))})",
                    list(row.values()),
                )
            db.commit()
        finally:
            db.close()
    return db_path


async def run_episode(
    task: Task,
    base_db: Path,
    agent: AgentFn,
    *,
    read_only: bool = False,
    timeout_s: float = 1200,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="episode-") as tmp:
        db_path = prepare_episode(base_db, task, Path(tmp))
        before = snapshot(db_path)
        store = PlantStore(db_path)
        error, run = None, None
        start = time.perf_counter()
        try:
            async with Toolkit(plant_servers(store, read_only=read_only)) as kit:
                try:
                    with anyio.fail_after(timeout_s):
                        run = await agent(task.question, kit)
                except TimeoutError:
                    error = f"timeout after {timeout_s:.0f} s"
                except Exception as exc:  # an LLM or transport failure ends this episode only
                    error = f"{type(exc).__name__}: {exc}"[:500]
                calls = list(kit.calls)
        finally:
            store.close()
        after = snapshot(db_path)
    final = run.answer if run else ""
    result = score(task, final, before, after)
    called = {c.name for c in calls}
    return {
        "task": task.id,
        "tier": task.tier,
        "template": task.template,
        "split": task.split,
        "source": task.source,
        "passed": result.passed and error is None,
        "reason": error or result.reason,
        "answer": result.answer,
        "format_ok": result.format_ok,
        "precision": result.precision,
        "recall": result.recall,
        "writes": result.writes,
        "expected": task.expected,
        "final": final[-2000:],
        "error": error,
        "stopped": run.stopped if run else "error",
        "steps": run.steps if run else 0,
        "seconds": round(time.perf_counter() - start, 2),
        "tokens_in": run.tokens_in if run else 0,
        "tokens_out": run.tokens_out if run else 0,
        "notes": run.notes if run else {},
        "tool_calls": len(calls),
        "tool_errors": sum(c.error is not None for c in calls),
        "tool_recall": (len(called & set(task.tools)) / len(task.tools)) if task.tools else None,
        "calls": [
            {"name": c.name, "args": c.arguments, "error": c.error, "seconds": round(c.seconds, 3)}
            for c in calls
        ],
    }


def make_agent(settings: Settings, design: str = "react", *, max_steps: int = 10) -> AgentFn:
    if design not in DESIGNS:
        raise ValueError(f"unknown agent design '{design}' ({', '.join(DESIGNS)})")
    run_design, llm = DESIGNS[design], make_llm(settings)

    async def agent(question: str, kit: Toolkit) -> Run:
        return await run_design(question, kit, llm, max_steps=max_steps)

    return agent


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def run_suite(
    tasks: Sequence[Task],
    base_db: Path,
    agent: AgentFn,
    out: Path,
    *,
    read_only: bool = False,
    meta: dict[str, Any] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_results(out)
    done = {r["task"] for r in rows}
    for task in tasks:
        if task.id in done:
            continue
        row = {**(meta or {}), **await run_episode(task, base_db, agent, read_only=read_only)}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        rows.append(row)
        if on_result:
            on_result(row)
    return rows
