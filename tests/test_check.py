from typing import Any

import pytest

from shopfloor_agent.eval.check import extract_answer, score_answer, score_state
from shopfloor_agent.eval.tasks import Task


def task(kind: str, expected: Any, **kw: Any) -> Task:
    return Task("T1", "aggregate", "t", "q", kind, expected, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Chiller 9 had 20.\nANSWER: 20", "20"),
        ("**ANSWER:** 20", "20"),
        ("answer = `RUL0015`.", "RUL0015"),
        ("ANSWER: 1\nchecking again...\nANSWER: 2", "2"),  # the last line counts
        ("The answer is 20.", None),
    ],
)
def test_extract_answer(text: str, value: str | None) -> None:
    assert extract_answer(text) == value


def test_numbers() -> None:
    assert score_answer(task("int", 20), "ANSWER: 20").passed
    assert score_answer(task("int", 1234), "ANSWER: 1,234 work orders").passed
    assert not score_answer(task("int", 20), "ANSWER: 21").passed
    # floats: two-decimal rounding is accepted even when the relative tolerance is tiny
    t = task("float", 0.01, tolerance=0.005)
    assert score_answer(t, "ANSWER: 0.0123").passed
    assert not score_answer(t, "ANSWER: 0.02").passed
    assert score_answer(task("float", 698.49, tolerance=0.005), "ANSWER: 698.4879").passed


def test_text_matches_whole_names_only() -> None:
    t = task("text", "Chiller 14", aliases=("CWC04014",))
    assert score_answer(t, "ANSWER: Chiller 14 (CWC04014)").passed
    assert score_answer(t, "ANSWER: cwc04014").passed
    assert not score_answer(task("text", "Chiller 1"), "ANSWER: Chiller 14").passed
    assert score_answer(task("text", "Oil Analysis"), "ANSWER: **Oil analysis.**").passed


def test_sets_and_counts() -> None:
    t = task("set", ["M003", "M010"])
    assert score_answer(t, "ANSWER: M010, M003").passed
    partial = score_answer(t, "ANSWER: M003")
    assert not partial.passed
    assert (partial.precision, partial.recall) == (1.0, 0.5)
    c = task("counts", {"WORK_ORDER": 2, "ALERT": 2, "ANOMALY": 0})
    assert score_answer(c, "ANSWER: WORK_ORDER=2, ALERT=2, ANOMALY=0").passed
    assert score_answer(c, "ANSWER: work orders: 2, alerts: 2, anomalies: 0").passed
    assert not score_answer(c, "ANSWER: WORK_ORDER=2, ALERT=2").passed  # zero must be stated


def test_unanswerable_and_format_failures() -> None:
    t = task("none", None)
    assert score_answer(t, "There is no such chiller.\nANSWER: none").passed
    assert not score_answer(t, "ANSWER: 12").passed
    missing = score_answer(task("int", 3), "It is 3.")
    assert not missing.passed
    assert not missing.format_ok


WO = {"equipment_id": "E1", "description": "Inspect condenser water flow",
      "component": "condenser", "primary_code": None, "work_type": "CM", "priority": 2,
      "status": "WAPPR"}  # fmt: skip


def test_read_tasks_must_not_write() -> None:
    before = {"WO1": dict(WO, status="INPRG")}
    after = {"WO1": dict(WO, status="CLOSE")}
    ok, why = score_state(task("int", 1), before, after)
    assert not ok
    assert "read-only" in why
    assert score_state(task("int", 1), before, before) == (True, "")


def test_action_create_and_close() -> None:
    spec = {"equipment_id": "E1", "work_type": "CM", "priority": 2, "status": "WAPPR",
            "description_contains": "flow"}  # fmt: skip
    create = Task("W1", "action", "create", "q", "action", {"created": [spec]})
    assert score_state(create, {}, {"WO9": WO}) == (True, "")
    assert not score_state(create, {}, {"WO9": dict(WO, priority=3)})[0]
    assert not score_state(create, {}, {"WO9": WO, "WO10": WO})[0]  # one too many
    nothing = Task("W2", "action", "cond", "q", "action", {"created": []})
    assert score_state(nothing, {}, {}) == (True, "")
    assert not score_state(nothing, {}, {"WO9": WO})[0]

    open_wos = {"WO1": dict(WO, status="INPRG"), "WO2": dict(WO, status="INPRG",
                                                             equipment_id="E2")}  # fmt: skip
    close = Task("W3", "action", "close", "q", "action",
                 {"status": {"WO1": "CLOSE", "WO2": "INPRG"}})  # fmt: skip
    assert score_state(close, open_wos, {**open_wos, "WO1": dict(WO, status="CLOSE")})[0]
    over = {"WO1": dict(WO, status="CLOSE"), "WO2": dict(WO, status="CLOSE", equipment_id="E2")}
    ok, why = score_state(close, open_wos, over)
    assert not ok
    assert "WO2" in why


def test_rescore_applies_a_stricter_rule_to_saved_rows() -> None:
    from shopfloor_agent.eval.check import rescore

    row = {"task": "T1", "passed": True, "error": None, "writes": [], "final": "ANSWER: MT001",
           "reason": "", "answer": "MT001", "format_ok": True}  # fmt: skip
    lenient = task("text", "Routine Maintenance", aliases=("MT001",))
    strict = task("text", "Routine Maintenance")
    assert rescore(lenient, row)["passed"]
    out = rescore(strict, row)
    assert not out["passed"]
    assert "expected Routine Maintenance" in out["reason"]
    assert not rescore(strict, {**row, "final": "ANSWER: Routine Maintenance",
                                "writes": ["created WO1"]})["passed"]  # fmt: skip
    action = Task("W1", "action", "a", "q", "action", {"created": []})
    assert rescore(action, {**row, "passed": False}) == {**row, "passed": False}
