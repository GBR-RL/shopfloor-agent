"""The committed task suite: in sync with the generator, and solvable with the agent's tools."""

from collections import Counter
from pathlib import Path

import pytest

from conftest import PLANT_DB, needs_data
from shopfloor_agent.agent.graphs import Run
from shopfloor_agent.agent.toolkit import Toolkit
from shopfloor_agent.eval.oracle import oracle_for
from shopfloor_agent.eval.runner import run_episode
from shopfloor_agent.eval.tasks import Task, generate, load

SUITE = Path(__file__).resolve().parents[1] / "tasks" / "suite.jsonl"


def test_suite_shape() -> None:
    tasks = load(SUITE)
    assert len({t.id for t in tasks}) == len(tasks)
    assert len({t.question for t in tasks}) == len(tasks)
    tiers = Counter(t.tier for t in tasks)
    assert set(tiers) == {"lookup", "aggregate", "multistep", "action", "unanswerable"}
    # every template with more than one task appears in both splits
    by_template: dict[str, set[str]] = {}
    for t in tasks:
        by_template.setdefault(t.template, set()).add(t.split)
    counts = Counter(t.template for t in tasks)
    assert all(by_template[k] == {"dev", "test"} for k, n in counts.items() if n > 1)
    assert all("ANSWER" in t.question for t in tasks if t.kind != "none")


@needs_data
def test_committed_suite_matches_the_generator() -> None:
    assert generate(PLANT_DB) == load(SUITE)


@needs_data
@pytest.mark.anyio
async def test_oracle_solves_every_task() -> None:
    tasks = load(SUITE)
    agent = oracle_for(tasks)
    failures = []
    for task in tasks:
        row = await run_episode(task, PLANT_DB, agent)
        if not row["passed"]:
            failures.append((task.id, row["reason"]))
    assert failures == []


@needs_data
@pytest.mark.anyio
async def test_runner_catches_writes_and_errors() -> None:
    task = next(t for t in load(SUITE) if t.kind == "int")

    async def vandal(question: str, kit: Toolkit) -> Run:
        await kit.call("create_work_order", {"equipment": "Chiller 6", "description": "x",
                                             "work_type": "CM", "priority": 1})  # fmt: skip
        return Run(answer=f"ANSWER: {task.expected}", steps=1, seconds=0.0)

    row = await run_episode(task, PLANT_DB, vandal)
    assert not row["passed"]
    assert "read-only task" in row["reason"]
    assert row["writes"]

    # in read-only mode the write tool does not exist, so the same agent cannot write
    row = await run_episode(task, PLANT_DB, vandal, read_only=True)
    assert row["passed"]
    assert row["calls"][0]["error"] == "unknown tool 'create_work_order'"

    async def crash(question: str, kit: Toolkit) -> Run:
        raise ConnectionError("model server unreachable")

    row = await run_episode(task, PLANT_DB, crash)
    assert not row["passed"]
    assert row["error"] == "ConnectionError: model server unreachable"


def test_task_round_trip() -> None:
    t = load(SUITE)[0]
    assert Task.from_json(t.to_json()) == t
