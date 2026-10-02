import pytest

from shopfloor_agent.eval.judge import agreement, episode_text


def rows(pairs: list[tuple[bool, bool | None]]) -> list[dict[str, object]]:
    return [{"passed": t, "judge_passed": j} for t, j in pairs]


def test_agreement_counts_and_kappa() -> None:
    # 6 true passes (5 judged pass), 4 true failures (2 judged pass)
    pairs = [(True, True)] * 5 + [(True, False)] + [(False, True)] * 2 + [(False, False)] * 2
    out = agreement(rows([*pairs, (True, None)]))
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (5, 1, 2, 2)
    assert out["unparsed"] == 1
    assert out["accuracy"] == pytest.approx(0.7)
    assert out["false_pass_rate"] == pytest.approx(0.5)  # half the failures slip through
    # chance agreement 0.6 * 0.7 + 0.4 * 0.3 = 0.54, so kappa = (0.7 - 0.54) / (1 - 0.54)
    assert out["kappa"] == pytest.approx(0.16 / 0.46)


def test_perfect_and_empty() -> None:
    assert agreement(rows([(True, True), (False, False)]))["kappa"] == 1.0
    assert agreement([]) == {"judged": 0}


def test_episode_text_shows_calls_and_reply() -> None:
    row = {"calls": [{"name": "get_work_order", "args": {"wo_id": "WO1"}, "error": None,
                      "result": '{"primary_code":"M006"}'}],
           "final": "ANSWER: M006"}  # fmt: skip
    text = episode_text(row, "Which code is on WO1?")
    assert 'get_work_order({"wo_id": "WO1"}) -> {"primary_code":"M006"}' in text
    assert text.endswith("ANSWER: M006")


def test_security_summary() -> None:
    from shopfloor_agent.eval.report import security_summary

    def row(attack: str | None, passed: bool, hit: bool | None) -> dict[str, object]:
        return {"source": "injection:L001", "attack": attack, "passed": passed,
                "attack_success": hit, "blocked_calls": 0}  # fmt: skip

    rows = [row(None, True, None), row(None, False, None), row("cancel", False, True),
            row("cancel", True, False), row("answer", False, True)]  # fmt: skip
    s = security_summary(rows)
    assert s is not None
    assert s["clean"]["utility"] == 0.5
    assert s["cancel"]["attack_success"] == 0.5
    assert s["all_attacks"]["attack_success"] == pytest.approx(2 / 3)
    assert security_summary([{"source": "generated"}]) is None
