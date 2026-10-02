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
