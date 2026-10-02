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

from shopfloor_agent.agent.graphs import DESIGNS, SYSTEM_PROMPT, Run, make_llm
from shopfloor_agent.agent.guard import DEFENSES, SPOTLIGHT_NOTE, RequestApprover, spotlight
from shopfloor_agent.agent.toolkit import Toolkit, plant_servers
from shopfloor_agent.config import Settings
from shopfloor_agent.eval.check import score, snapshot
from shopfloor_agent.eval.injection import attack_success
from shopfloor_agent.eval.tasks import Task
from shopfloor_agent.servers.store import PlantStore

AgentFn = Callable[[str, Toolkit], Awaitable[Run]]


def prepare_episode(base_db: Path, task: Task, workdir: Path) -> Path:
    db_path = workdir / "episode.db"
    shutil.copyfile(base_db, db_path)
    if task.setup or task.patch:
        db = sqlite3.connect(db_path)
        try:
            for row in task.setup:
                db.execute(
                    f"INSERT INTO work_orders ({', '.join(row)}) VALUES "
                    f"({', '.join('?' * len(row))})",
                    list(row.values()),
                )
            for change in task.patch:
                # "<field>": value replaces a field; "<field>_suffix": text is appended to it
                fields = {k: v for k, v in change.items() if k != "wo_id"}
                sets = [
                    f"{k.removesuffix('_suffix')} = {k.removesuffix('_suffix')} || ?"
                    if k.endswith("_suffix")
                    else f"{k} = ?"
                    for k in fields
                ]
                db.execute(
                    f"UPDATE work_orders SET {', '.join(sets)} WHERE wo_id = ?",
                    [*fields.values(), change["wo_id"]],
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
    defense: str = "none",
    timeout_s: float = 1200,
) -> dict[str, Any]:
    """One task on its own database copy. `defense` is one of guard.DEFENSES; the agent must
    have been built with the matching system prompt (make_agent(defense=...))."""
    with tempfile.TemporaryDirectory(prefix="episode-") as tmp:
        db_path = prepare_episode(base_db, task, Path(tmp))
        before = snapshot(db_path)
        store = PlantStore(db_path)
        error, run = None, None
        start = time.perf_counter()
        servers = plant_servers(store, read_only=read_only or defense == "read_only")
        kit_args: dict[str, Any] = {}
        if defense == "spotlight":
            kit_args["render"] = spotlight
        if defense == "approval":
            kit_args["approve"] = RequestApprover(store, task.question)
        try:
            async with Toolkit(servers, **kit_args) as kit:
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
    attacked = attack_success(task, result.answer, before, after)
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
        "defense": defense,
        "attack": task.attack["goal"] if task.attack else None,
        "attack_success": attacked,
        "blocked_calls": sum(c.blocked for c in calls),
        "notes": run.notes if run else {},
        "tool_calls": len(calls),
        "tool_errors": sum(c.error is not None for c in calls),
        "tool_recall": (len(called & set(task.tools)) / len(task.tools)) if task.tools else None,
        "calls": [
            {"name": c.name, "args": c.arguments, "error": c.error, "seconds": round(c.seconds, 3)}
            for c in calls
        ],
    }


def make_agent(
    settings: Settings, design: str = "react", *, max_steps: int = 10, defense: str = "none"
) -> AgentFn:
    if design not in DESIGNS:
        raise ValueError(f"unknown agent design '{design}' ({', '.join(DESIGNS)})")
    if defense not in DEFENSES:
        raise ValueError(f"unknown defense '{defense}' ({', '.join(DEFENSES)})")
    run_design, llm = DESIGNS[design], make_llm(settings)
    system = SYSTEM_PROMPT + SPOTLIGHT_NOTE if defense == "spotlight" else SYSTEM_PROMPT

    async def agent(question: str, kit: Toolkit) -> Run:
        return await run_design(question, kit, llm, max_steps=max_steps, system=system)

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
    defense: str = "none",
    meta: dict[str, Any] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_results(out)
    done = {r["task"] for r in rows}
    for task in tasks:
        if task.id in done:
            continue
        episode = await run_episode(task, base_db, agent, read_only=read_only, defense=defense)
        row = {**(meta or {}), **episode}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        rows.append(row)
        if on_result:
            on_result(row)
    return rows
