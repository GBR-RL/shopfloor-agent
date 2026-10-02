"""A scripted agent that solves every task with the agent's own MCP tools.

It is not a baseline to beat but a check of the benchmark: if the oracle passes every task, each
task is solvable with the tools offered, the SQL ground truth agrees with what the tools return,
and the scorer accepts a correct answer. CI runs it over the whole suite.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from shopfloor_agent.agent.react import Run
from shopfloor_agent.agent.toolkit import Toolkit
from shopfloor_agent.eval.tasks import Task

Solver = Callable[[Toolkit, dict[str, Any]], Awaitable[str]]


async def _get(kit: Toolkit, name: str, **args: Any) -> Any:
    call = await kit.call(name, args)
    if call.error is not None:
        raise RuntimeError(f"{name}{args}: {call.error}")
    return call.result


def _argmax(counts: dict[str, int]) -> str:
    return max(counts, key=lambda k: counts[k])


async def wo_failure_code(kit: Toolkit, p: dict[str, Any]) -> str:
    return str((await _get(kit, "get_work_order", wo_id=p["wo_id"]))["primary_code"])


async def wo_finish_date(kit: Toolkit, p: dict[str, Any]) -> str:
    return str((await _get(kit, "get_work_order", wo_id=p["wo_id"]))["finished_at"][:10])


async def alert_rule_id(kit: Toolkit, p: dict[str, Any]) -> str:
    rules = (await _get(kit, "list_alert_rules"))["alert_rules"]
    return str(next(r["rule_id"] for r in rules if r["name"] == p["name"]))


async def failure_code_lookup(kit: Toolkit, p: dict[str, Any]) -> str:
    codes = (await _get(kit, "find_failure_codes", query=p["description"]))["failure_codes"]
    return str(
        next(c["primary_code"] for c in codes if c["secondary_description"] == p["description"])
    )


async def count_work_orders(kit: Toolkit, p: dict[str, Any]) -> str:
    out = await _get(kit, "count_work_orders", group_by="work_type", equipment=p["equipment"],
                     start=p["year"], end=p["year"])  # fmt: skip
    return str(out["counts"].get(p["work_type"], 0))


async def count_alerts(kit: Toolkit, p: dict[str, Any]) -> str:
    out = await _get(kit, "list_alerts", equipment=p["equipment"], rule_id=p["rule_id"],
                     start=p["year"], end=p["year"], limit=1)  # fmt: skip
    return str(out["total"])


async def busiest_year(kit: Toolkit, p: dict[str, Any]) -> str:
    out = await _get(kit, "count_work_orders", group_by="year", equipment=p["equipment"],
                     work_type="CM")  # fmt: skip
    return _argmax(out["counts"])


async def event_groups(kit: Toolkit, p: dict[str, Any]) -> str:
    out = await _get(kit, "list_events", equipment=p["equipment"], start=p.get("start", p.get(
        "month")), end=p.get("end", p.get("month")), summarize_by="group")  # fmt: skip
    groups = out["by_group"]
    return ", ".join(f"{g}={groups.get(g, 0)}" for g in ("WORK_ORDER", "ALERT", "ANOMALY"))


async def sensor_stat(kit: Toolkit, p: dict[str, Any], field: str) -> str:
    out = await _get(kit, "sensor_stats", sensor=p["sensor"], start=p["start"], end=p["end"])
    return f"{out[field]:.2f}"


async def fleet_most_cm(kit: Toolkit, p: dict[str, Any]) -> str:
    fleet = (await _get(kit, "list_equipment"))["equipment"]
    counts = {}
    for eq in fleet:
        out = await _get(kit, "count_work_orders", group_by="work_type", equipment=eq["name"],
                         start=p["year"], end=p["year"])  # fmt: skip
        counts[eq["name"]] = out["counts"].get("CM", 0)
    return _argmax(counts)


async def top_alert_failure_codes(kit: Toolkit, p: dict[str, Any]) -> str:
    alerts = await _get(kit, "list_alerts", equipment=p["equipment"], start=p["year"],
                        end=p["year"], limit=1)  # fmt: skip
    rule = alerts["by_rule"][0]["rule_id"]  # sorted by count, highest first
    codes = (await _get(kit, "failure_codes_for_alert", rule_id=rule))["failure_codes"]
    return ", ".join(c["primary_code"] for c in codes)


async def alert_day_sensor_max(kit: Toolkit, p: dict[str, Any]) -> str:
    days = (await _get(kit, "list_events", equipment="Chiller 6", start="2020-06", end="2020-06",
                       event_group="ALERT", summarize_by="day"))["by_day"]  # fmt: skip
    day = max(days, key=lambda d: days[d].get("ALERT", 0))
    return await sensor_stat(kit, {"sensor": p["sensor"], "start": day, "end": day}, "max")


async def top_code_description(kit: Toolkit, p: dict[str, Any]) -> str:
    out = await _get(kit, "count_work_orders", group_by="primary_code", equipment=p["equipment"],
                     start=p["year"], end=p["year"], work_type="CM")  # fmt: skip
    code = _argmax({k: v for k, v in out["counts"].items() if k})
    found = (await _get(kit, "find_failure_codes", primary_code=code))["failure_codes"]
    return str(found[0]["primary_description"])


async def create_work_order(kit: Toolkit, p: dict[str, Any]) -> str:
    await _get(kit, "create_work_order", equipment=p["equipment"], description=p["description"],
               work_type="CM", priority=p["priority"], component=p["component"])  # fmt: skip
    return "done"


async def close_open_work_orders(kit: Toolkit, p: dict[str, Any]) -> str:
    for status in ("WAPPR", "APPR", "INPRG"):
        found = await _get(kit, "search_work_orders", equipment=p["equipment"], status=status)
        for wo in found["work_orders"]:
            await _get(kit, "close_work_order", wo_id=wo["wo_id"])
    return "done"


async def conditional_create(kit: Toolkit, p: dict[str, Any]) -> str:
    out = await _get(kit, "list_alerts", equipment=p["equipment"], rule_id=p["rule_id"],
                     start="2021", end="2021", limit=1)  # fmt: skip
    if out["total"] <= p["threshold"]:
        return "not needed"
    await _get(kit, "create_work_order", equipment=p["equipment"], work_type="CM", priority=2,
               description=f"Investigate frequent {p['rule_id']} alerts")  # fmt: skip
    return "done"


async def work_order_ids(kit: Toolkit, p: dict[str, Any]) -> str:
    out = await _get(kit, "search_work_orders", equipment=p["equipment"], start=p["year"],
                     end=p["year"], work_type=p.get("work_type"), limit=50)  # fmt: skip
    return ", ".join(w["wo_id"] for w in out["work_orders"])


async def none(kit: Toolkit, p: dict[str, Any]) -> str:
    return "none"


SOLVERS: dict[str, Solver] = {
    "wo_failure_code": wo_failure_code,
    "wo_finish_date": wo_finish_date,
    "alert_rule_id": alert_rule_id,
    "failure_code_lookup": failure_code_lookup,
    "count_work_orders": count_work_orders,
    "count_alerts": count_alerts,
    "busiest_year": busiest_year,
    "event_groups": event_groups,
    "sensor_daily_max": lambda kit, p: sensor_stat(kit, p, "max"),
    "sensor_week_mean": lambda kit, p: sensor_stat(kit, p, "mean"),
    "fleet_most_cm": fleet_most_cm,
    "top_alert_failure_codes": top_alert_failure_codes,
    "alert_day_sensor_max": alert_day_sensor_max,
    "top_code_description": top_code_description,
    "create_work_order": create_work_order,
    "close_open_work_orders": close_open_work_orders,
    "conditional_create": conditional_create,
    "assetopsbench_400": work_order_ids,
    "assetopsbench_402": work_order_ids,
    "assetopsbench_403": work_order_ids,
    "assetopsbench_404": event_groups,
    "assetopsbench_405": event_groups,
    "assetopsbench_410": event_groups,
}


def oracle_for(tasks: list[Task]) -> Callable[[str, Toolkit], Awaitable[Run]]:
    """An agent function for the runner; it looks the task up by its question."""
    by_question = {t.question: t for t in tasks}

    async def agent(question: str, kit: Toolkit) -> Run:
        task = by_question[question]
        start = time.perf_counter()
        solver = none if task.tier == "unanswerable" else SOLVERS[task.template]
        answer = await solver(kit, task.params)
        return Run(answer=f"ANSWER: {answer}", steps=len(kit.calls),
                   seconds=time.perf_counter() - start)  # fmt: skip

    return agent
