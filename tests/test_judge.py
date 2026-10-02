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
