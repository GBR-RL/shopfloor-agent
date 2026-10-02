"""The routed design: dispatch per route, one shared budget, and a safe fallback."""

from typing import Any

import pytest

from shopfloor_agent.agent import graphs
from shopfloor_agent.agent.graphs import Run, run_routed
from shopfloor_agent.eval.report import routing_summary

pytestmark = pytest.mark.anyio


def stub(name: str, seen: dict[str, Any]) -> Any:
    async def design(question: str, kit: Any, llm: Any, *, max_steps: int, system: str) -> Run:
        seen.update(design=name, max_steps=max_steps)
        return Run(answer="ANSWER: 1", steps=3, seconds=0.0, tokens_in=100, tokens_out=10,
                   notes={"plan": ["x"]} if name == "plan" else {})  # fmt: skip

    return design


@pytest.mark.parametrize("route", ["plan", "react"])
async def test_dispatches_on_the_route(monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    seen: dict[str, Any] = {}

    async def choose(llm: Any, question: str) -> tuple[str, str, None]:
        return route, "because", None

    monkeypatch.setattr(graphs, "choose_route", choose)
    monkeypatch.setattr(graphs, "run_plan_execute", stub("plan", seen))
    monkeypatch.setattr(graphs, "run_react", stub("react", seen))
    run = await run_routed("q", kit=None, llm=None, max_steps=16)  # type: ignore[arg-type]
    assert seen["design"] == route
    assert seen["max_steps"] == 15  # the router call comes out of the same budget
    assert run.steps == 4
    assert run.notes["route"] == route
    assert run.notes["route_reason"] == "because"


async def test_router_failure_falls_back_to_react() -> None:
    class Broken:
        def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
            class Runnable:
                async def ainvoke(self, messages: Any) -> Any:
                    raise ConnectionError("model server down")

            return Runnable()

    route, reason, raw = await graphs.choose_route(Broken(), "q")  # type: ignore[arg-type]
    assert (route, raw) == ("react", None)
    assert "ConnectionError" in reason


def test_routing_summary() -> None:
    def row(tier: str, route: str | None) -> dict[str, Any]:
        return {"tier": tier, "notes": {"route": route} if route else {}}

    rows = [row("multistep", "plan"), row("multistep", "react"), row("lookup", "react"),
            row("lookup", "plan"), row("action", None)]  # fmt: skip
    s = routing_summary(rows)
    assert s is not None
    assert s["multistep"] == {"plan": 1, "react": 1}
    assert s["lookup"] == {"plan": 1, "react": 1}
    assert routing_summary([row("lookup", None)]) is None
