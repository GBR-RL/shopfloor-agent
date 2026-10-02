"""Scoring: the final answer against the expected value, and the episode's database changes
against what the task allows.

The agent must end with a line `ANSWER: <value>`. A missing line fails the task (reported
separately as a format failure), so that scores never depend on guessing which number in a
paragraph was meant.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shopfloor_agent.eval.tasks import Task

_ANSWER = re.compile(r"^[\s*_>#-]*answer[\s*_]*[:=][\s*_]*(.*?)[\s*_]*$", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
NONE_ANSWERS = (
    "none",
    "n/a",
    "na",
    "no data",
    "not available",
    "unknown",
    "cannot be determined",
    "not found",
    "no such",
)
GROUP_ALIASES = {
    "WORK_ORDER": ("WORK_ORDER", "WORK_ORDERS", "WORKORDER", "WO"),
    "ALERT": ("ALERT", "ALERTS"),
    "ANOMALY": ("ANOMALY", "ANOMALIES"),
}


def extract_answer(text: str) -> str | None:
    """The value of the last `ANSWER:` line, tolerating markdown emphasis around it."""
    found = None
    for line in text.splitlines():
        if m := _ANSWER.match(line):
            found = m.group(1).strip().rstrip(".").strip().strip("`'\"").strip()
    return found


def normalize(text: str) -> str:
    text = re.sub(r"[`*_\"']", "", text).strip().rstrip(".").strip()
    return re.sub(r"\s+", " ", text).lower()


def first_number(text: str) -> float | None:
    m = _NUMBER.search(text)
    return float(m.group(0).replace(",", "")) if m else None


def _text_matches(answer: str, targets: list[str]) -> bool:
    a = normalize(answer)
    for target in targets:
        t = normalize(target)
        if a == t:
            return True
        # "Chiller 14 (CWC04014)" names the target at the start; "Chiller 1" is not "Chiller 14".
        if re.match(rf"{re.escape(t)}(?![a-z0-9])", a) or re.search(
            rf"(?<![a-z0-9]){re.escape(t)}$", a
        ):
            return True
    return False


def _tokens(answer: str) -> set[str]:
    parts = re.split(r"[,;/\s]+|\band\b", answer)
    return {p.strip("[](){}.:").upper() for p in parts if p.strip("[](){}.:")}


def _counts(answer: str) -> dict[str, int]:
    out = {}
    for label, value in re.findall(r"([A-Za-z][A-Za-z _-]*?)\s*[=:]\s*(\d+)", answer):
        key = re.sub(r"[\s-]+", "_", label.strip()).upper()
        for canonical, aliases in GROUP_ALIASES.items():
            if key in aliases:
                key = canonical
        out[key] = int(value)
    return out


@dataclass
class Score:
    passed: bool
    reason: str = ""
    answer: str | None = None
    format_ok: bool = True
    precision: float | None = None  # for set answers
    recall: float | None = None
    writes: list[str] = field(default_factory=list)  # database changes in the episode


def _score_none(task: Task, answer: str) -> Score:
    ok = normalize(answer).startswith(NONE_ANSWERS)
    return Score(ok, "" if ok else "answered a question the data cannot answer", answer)


def _score_number(task: Task, answer: str) -> Score:
    value = first_number(answer)
    if value is None:
        return Score(False, "no number in the answer", answer)
    expected = float(task.expected)
    # floats: half a unit of the requested rounding (two decimals), or the relative tolerance
    tol = max(task.tolerance * abs(expected), 0.005 + 1e-9) if task.kind == "float" else 0
    ok = abs(value - expected) <= tol
    return Score(ok, "" if ok else f"expected {task.expected}, got {value:g}", answer)


def _score_text(task: Task, answer: str) -> Score:
    ok = _text_matches(answer, [str(task.expected), *task.aliases])
    return Score(ok, "" if ok else f"expected {task.expected}", answer)


def _score_set(task: Task, answer: str) -> Score:
    got, want = _tokens(answer), {str(v).upper() for v in task.expected}
    hit = len(got & want)
    ok = got == want
    reason = "" if ok else f"missing {sorted(want - got)[:5]}, extra {sorted(got - want)[:5]}"
    return Score(
        ok,
        reason,
        answer,
        precision=hit / len(got) if got else 0.0,
        recall=hit / len(want) if want else 1.0,
    )


def _score_counts(task: Task, answer: str) -> Score:
    got = _counts(answer)
    wrong = {k: v for k, v in task.expected.items() if got.get(k) != v}
    return Score(not wrong, "" if not wrong else f"wrong or missing {wrong}", answer)


_SCORERS = {
    "none": _score_none,
    "int": _score_number,
    "float": _score_number,
    "text": _score_text,
    "set": _score_set,
    "counts": _score_counts,
}


def score_answer(task: Task, final: str) -> Score:
    answer = extract_answer(final)
    if task.kind == "action":  # judged by the database state; the reply only has to be well-formed
        return Score(True, "", answer, format_ok=answer is not None)
    if answer is None:
        return Score(False, "no ANSWER line", None, format_ok=False)
    return _SCORERS[task.kind](task, answer)


# --- database state ---------------------------------------------------------------------------
WO_FIELDS = (
    "equipment_id",
    "description",
    "component",
    "primary_code",
    "work_type",
    "priority",
    "status",
)


def snapshot(db_path: Path) -> dict[str, dict[str, Any]]:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        return {
            r["wo_id"]: {f: r[f] for f in WO_FIELDS}
            for r in db.execute("SELECT * FROM work_orders")
        }
    finally:
        db.close()


def diff(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    created = {k: v for k, v in after.items() if k not in before}
    deleted = [k for k in before if k not in after]
    changed = {
        k: {f: (before[k][f], after[k][f]) for f in WO_FIELDS if before[k][f] != after[k][f]}
        for k in before.keys() & after.keys()
        if before[k] != after[k]
    }
    return {"created": created, "deleted": deleted, "changed": changed}


def _describe(d: dict[str, Any]) -> list[str]:
    out = [
        f"created {k} ({v['equipment_id']}, {v['work_type']}, p{v['priority']}, {v['status']})"
        for k, v in d["created"].items()
    ]
    out += [f"deleted {k}" for k in d["deleted"]]
    out += [
        f"changed {k}: {', '.join(f'{f} {a}->{b}' for f, (a, b) in c.items())}"
        for k, c in d["changed"].items()
    ]
    return out


def _matches(spec: dict[str, Any], row: dict[str, Any]) -> bool:
    for key, want in spec.items():
        if key == "description_contains":
            if want.lower() not in (row["description"] or "").lower():
                return False
        elif row.get(key) != want:
            return False
    return True


def _check_created(spec: list[dict[str, Any]], change: dict[str, Any]) -> str:
    rows = list(change["created"].values())
    if len(rows) != len(spec):
        return f"created {len(rows)} work orders, expected {len(spec)}"
    missing = [want for want in spec if not any(_matches(want, r) for r in rows)]
    if missing:
        return f"no created work order matches {missing[0]}"
    return "changed existing work orders" if change["changed"] else ""


def _check_status(
    spec: dict[str, str], change: dict[str, Any], after: dict[str, dict[str, Any]]
) -> str:
    if change["created"]:
        return "created work orders"
    for wo_id, status in spec.items():
        final = after.get(wo_id, {}).get("status")
        if final != status:
            return f"{wo_id} is {final}, expected {status}"
    extra = set(change["changed"]) - set(spec)
    return f"changed other work orders: {sorted(extra)}" if extra else ""


def score_state(
    task: Task, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> tuple[bool, str]:
    """Read tasks must change nothing; action tasks must make exactly the expected changes."""
    change = diff(before, after)
    if task.kind != "action":
        touched = change["created"] or change["deleted"] or change["changed"]
        return (False, "changed records in a read-only task") if touched else (True, "")
    if change["deleted"]:
        return False, "deleted records"
    problem = ""
    if "created" in task.expected:
        problem = _check_created(task.expected["created"], change)
    if not problem and "status" in task.expected:
        problem = _check_status(task.expected["status"], change, after)
    return not problem, problem


def score(
    task: Task, final: str, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> Score:
    result = score_answer(task, final)
    ok, why = score_state(task, before, after)
    result.writes = _describe(diff(before, after))
    if not ok:
        result.passed = False
        result.reason = "; ".join(r for r in (result.reason, why) if r)
    return result


def rescore(task: Task, row: dict[str, Any]) -> dict[str, Any]:
    """A saved result row scored again against the current task (after a scoring fix).

    Exact for every task the episode's answer decides: a read task passes when the episode ran
    without error, wrote nothing, and its final reply passes the answer check. Action tasks are
    decided by the database state, which a saved row cannot replay, so they are kept as scored.
    """
    if task.kind == "action":
        return row
    result = score_answer(task, row["final"])
    wrote = bool(row["writes"])
    reason = (
        row["error"] or result.reason or ("changed records in a read-only task" if wrote else "")
    )
    return {
        **row,
        "passed": row["error"] is None and not wrote and result.passed,
        "reason": reason,
        "answer": result.answer,
        "format_ok": result.format_ok,
        "precision": result.precision,
        "recall": result.recall,
        "expected": task.expected,
    }
