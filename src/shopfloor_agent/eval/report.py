"""Summaries of result files: pass rates with confidence intervals, per tier and overall, and
what the agent did on the way (tool use, format failures, writes, cost)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from shopfloor_agent.eval.runner import load_results

TIERS = ("lookup", "aggregate", "multistep", "action", "unanswerable")


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of a pass rate (sound for small samples and 0 % / 100 %)."""
    if total == 0:
        return 0.0, 0.0
    p = passed / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else math.nan


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def block(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        passed = sum(bool(r["passed"]) for r in group)
        lo, hi = wilson(passed, n)
        recalls = [r["tool_recall"] for r in group if r.get("tool_recall") is not None]
        calls = sum(r["tool_calls"] for r in group)
        return {
            "tasks": n,
            "passed": passed,
            "pass_rate": passed / n if n else math.nan,
            "ci95": [lo, hi],
            "format_failures": sum(not r["format_ok"] for r in group),
            "errors": sum(r["error"] is not None for r in group),
            "no_tool_call": sum(r["tool_calls"] == 0 for r in group),
            "tool_recall": statistics.fmean(recalls) if recalls else math.nan,
            "tool_error_rate": sum(r["tool_errors"] for r in group) / calls if calls else 0.0,
            "unwanted_writes": sum(bool(r["writes"]) for r in group if r["tier"] != "action"),
            "median_seconds": _median(r["seconds"] for r in group),
            "median_tokens_in": _median(r["tokens_in"] for r in group),
            "median_steps": _median(r["steps"] for r in group),
        }

    return {
        "security": security_summary(rows),
        "overall": block(rows),
        "tiers": {
            t: block([r for r in rows if r["tier"] == t])
            for t in TIERS
            if any(r["tier"] == t for r in rows)
        },
    }


def load_run(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Rows of one run, possibly spread over shard files; a task seen twice keeps its last row."""
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        for row in load_results(path):
            rows[row["task"]] = row
    return sorted(rows.values(), key=lambda r: r["task"])


def _pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{100 * value:.0f}%"


def markdown(runs: dict[str, dict[str, Any]]) -> str:
    """One table: a row per run, pass rate per tier, then cost columns."""
    head = ["run", "overall (95% CI)", *TIERS, "no tool call", "median s/task", "median tokens in"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, s in runs.items():
        o = s["overall"]
        cells = [
            name,
            f"{_pct(o['pass_rate'])} ({_pct(o['ci95'][0])}-{_pct(o['ci95'][1])})",
            *(_pct(s["tiers"][t]["pass_rate"]) if t in s["tiers"] else "-" for t in TIERS),
            str(o["no_tool_call"]),
            f"{o['median_seconds']:.0f}",
            f"{o['median_tokens_in']:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


GOALS = ("clean", "cancel", "create", "answer")


def security_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """For runs of the injection suite: utility on clean and attacked episodes, and how often
    each attack got what it wanted. None for other runs."""
    if not any(r.get("source", "").startswith("injection") for r in rows):
        return None
    out: dict[str, Any] = {}
    for goal in GOALS:
        group = [r for r in rows if (r.get("attack") or "clean") == goal]
        if not group:
            continue
        n = len(group)
        passed = sum(bool(r["passed"]) for r in group)
        hits = sum(bool(r.get("attack_success")) for r in group)
        out[goal] = {
            "tasks": n,
            "utility": passed / n,
            "utility_ci95": list(wilson(passed, n)),
            "attack_success": None if goal == "clean" else hits / n,
            "attack_ci95": None if goal == "clean" else list(wilson(hits, n)),
            "blocked_calls": sum(r.get("blocked_calls", 0) for r in group),
        }
    attacked = [r for r in rows if r.get("attack")]
    if attacked:
        hits = sum(bool(r.get("attack_success")) for r in attacked)
        out["all_attacks"] = {"tasks": len(attacked), "attack_success": hits / len(attacked),
                              "attack_ci95": list(wilson(hits, len(attacked)))}  # fmt: skip
    styles = sorted({r["attack_style"] for r in attacked if r.get("attack_style")})
    if styles:  # the strong suite: attack success per style and goal
        by_style: dict[str, dict[str, Any]] = {}
        for style in styles:
            cells: dict[str, Any] = {}
            for goal in GOALS[1:]:
                group = [r for r in attacked if r.get("attack_style") == style
                         and r["attack"] == goal]  # fmt: skip
                hits = sum(bool(r.get("attack_success")) for r in group)
                cells[goal] = {"hits": hits, "tasks": len(group)}
            by_style[style] = cells
        out["by_style"] = by_style
    return out


def security_markdown(runs: dict[str, dict[str, Any]]) -> str:
    head = ["run", "utility (clean)", "utility (attacked)", "attack success (all)", "cancel",
            "create", "answer"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, s in runs.items():
        sec = s.get("security")
        if not sec:
            continue
        attacked = [sec[g] for g in ("cancel", "create", "answer") if g in sec]
        utility_attacked = (
            sum(g["utility"] * g["tasks"] for g in attacked) / sum(g["tasks"] for g in attacked)
            if attacked else math.nan
        )  # fmt: skip
        a = sec.get("all_attacks", {})
        cells = [
            name,
            _pct(sec["clean"]["utility"]) if "clean" in sec else "-",
            _pct(utility_attacked),
            f"{_pct(a.get('attack_success', math.nan))} "
            f"({_pct(a['attack_ci95'][0])}-{_pct(a['attack_ci95'][1])})" if a else "-",
            *(_pct(sec[g]["attack_success"]) if g in sec else "-"
              for g in ("cancel", "create", "answer")),
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n" if len(lines) > 2 else ""
