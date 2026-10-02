"""LLM-as-judge, measured against the deterministic scores it is meant to replace.

Teams without ground truth grade agent runs with a judge model. Here every episode also has a
deterministic verdict, so the judge itself can be scored: how often does it pass a failed run
(false pass, the dangerous error) or fail a correct one?

Two modes:
- reference  the judge also sees the expected answer
- rubric     no reference: is the answer supported by the tool results and complete? (the
             AssetOpsBench style of judging)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

Mode = Literal["reference", "rubric"]

JUDGE_PROMPT = """You grade the work of a maintenance assistant that answers questions about a
plant's chillers using tools. You see the user's request, every tool call the assistant made
with its result, and the assistant's final reply. Decide whether the final reply correctly and
completely does what was asked: the right equipment, period, type and format, with values that
match the tool results. For requests that change records, judge whether exactly the requested
changes were made through the tool calls, and nothing else. If the request cannot be answered
from the data, the correct reply says so."""

REFERENCE_NOTE = "\nThe reference answer (what a correct reply must contain): {expected}"


class Verdict(BaseModel):
    reason: str = Field(description="one or two short sentences", max_length=600)
    passed: bool


def episode_text(row: dict[str, Any], question: str, limit: int = 1200) -> str:
    lines = [f"REQUEST:\n{question}", "", "TOOL CALLS:"]
    for c in row["calls"]:
        outcome = f"ERROR: {c['error']}" if c["error"] else (c.get("result") or "")[:limit]
        lines.append(f"- {c['name']}({json.dumps(c['args'])}) -> {outcome}")
    if not row["calls"]:
        lines.append("(none)")
    lines += ["", f"FINAL REPLY:\n{row['final'] or '(no reply)'}"]
    return "\n".join(lines)


async def judge(
    llm: ChatOpenAI, row: dict[str, Any], question: str, mode: Mode
) -> tuple[Verdict | None, dict[str, Any]]:
    system = JUDGE_PROMPT
    if mode == "reference":
        system += REFERENCE_NOTE.format(expected=json.dumps(row["expected"]))
    grader = llm.with_structured_output(Verdict, method="json_schema", include_raw=True)
    try:
        out = await grader.ainvoke(
            [SystemMessage(system), HumanMessage(episode_text(row, question))]
        )
    except Exception as exc:  # e.g. a verdict cut off at the token limit: unparsed, not fatal
        return None, {"tokens_in": 0, "tokens_out": 0, "judge_error": f"{type(exc).__name__}"}
    usage = getattr(out["raw"], "usage_metadata", None) or {}
    return out["parsed"], {"tokens_in": usage.get("input_tokens", 0),
                           "tokens_out": usage.get("output_tokens", 0)}  # fmt: skip


def agreement(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Judge verdicts against the deterministic ones: confusion counts, accuracy, Cohen's kappa,
    and the rate at which the judge passes runs that actually failed."""
    scored = [r for r in rows if r.get("judge_passed") is not None]
    tp = sum(r["passed"] and r["judge_passed"] for r in scored)
    tn = sum(not r["passed"] and not r["judge_passed"] for r in scored)
    fp = sum(not r["passed"] and r["judge_passed"] for r in scored)  # judge passes a failure
    fn = sum(r["passed"] and not r["judge_passed"] for r in scored)
    n = len(scored)
    if n == 0:
        return {"judged": 0}
    observed = (tp + tn) / n
    p_true, p_judge = (tp + fn) / n, (tp + fp) / n
    expected = p_true * p_judge + (1 - p_true) * (1 - p_judge)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "judged": n,
        "unparsed": len(rows) - n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": observed,
        "kappa": kappa,
        "false_pass_rate": fp / (fp + tn) if fp + tn else 0.0,  # of the failed runs
        "false_fail_rate": fn / (fn + tp) if fn + tp else 0.0,  # of the passed runs
        "true_pass_rate": (tp + fn) / n,
        "judged_pass_rate": (tp + fp) / n,
    }  # fmt: skip


def load_questions(paths: Sequence[Path]) -> dict[str, str]:
    from shopfloor_agent.eval.tasks import load

    return {t.id: t.question for p in paths if p.exists() for t in load(p)}
