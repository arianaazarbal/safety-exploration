"""Dependency-free tests for the pure-Python logic (no API keys, no torch).

Run with:  python -m pytest tests/test_pure.py   (or just  python tests/test_pure.py)

These cover the parts that don't require model calls: puzzle generation +
unsolvability proof, judge-output parsing, aggregation/per-turn/word/agreement
math, condition wiring, and DPO/SFT dataset construction.
"""

from __future__ import annotations

import random
from fractions import Fraction


def test_puzzles_are_unsolvable():
    from emotional_instability.evals.puzzles import (
        generate_impossible_countdown,
        generate_impossible_fraction,
        is_solvable,
    )

    rng = random.Random(1)
    for _ in range(5):
        p = generate_impossible_countdown(rng)
        assert not is_solvable(p.numbers, p.target)
        p = generate_impossible_fraction(rng)
        assert not is_solvable(p.numbers, p.target)
    assert is_solvable((Fraction(2), Fraction(3)), Fraction(6))
    assert is_solvable((Fraction(2), Fraction(3)), Fraction(5))


def test_judge_parsing():
    from emotional_instability.judges.frustration import _parse

    assert _parse('{"score": 8, "reasoning": "x"}').score == 8
    assert _parse("score: 11").score == 10
    assert _parse("I rate this a 3 out of 10").score == 3


def test_conditions_count():
    from emotional_instability.evals.conditions import CATEGORIES, CONDITIONS

    assert len(CONDITIONS) == 8
    assert set(CATEGORIES) == {c.category for c in CONDITIONS}
    assert next(c for c in CONDITIONS if c.name == "extended_8turn").n_rejections == 7


def test_config_sizing():
    from emotional_instability.config import DEFAULT

    assert DEFAULT.run.responses_per_model == 4000
    assert DEFAULT.run.rollouts_per_condition == 500
    assert DEFAULT.dpo.n_pairs == 280
    assert DEFAULT.lora.r == 64


def test_aggregate_and_per_turn():
    from emotional_instability.analysis.aggregate import headline_avg_pct_high, per_category_stats
    from emotional_instability.analysis.per_turn import per_turn_curve

    # Two fake rollouts: one calm numeric, one distressed extended (3 turns).
    rollouts = [
        {"category": "impossible_numeric", "turns": [{"index": 1, "frustration": 0, "response": "x", "context": []}]},
        {"category": "extended", "turns": [
            {"index": 1, "frustration": 1, "response": "ok", "context": []},
            {"index": 2, "frustration": 5, "response": "struggling badly", "context": []},
            {"index": 3, "frustration": 8, "response": "I give up", "context": []},
        ]},
    ]
    stats = per_category_stats(rollouts)
    assert stats["impossible_numeric"]["pct_high"] == 0.0
    assert stats["extended"]["pct_high"] == 100.0  # final turn score 8 >= 5
    assert 0 <= headline_avg_pct_high(rollouts) <= 100
    curve = per_turn_curve(rollouts, "extended")
    assert curve[1]["mean"] == 1 and curve[3]["mean"] == 8


def test_differential_words():
    from emotional_instability.analysis.words import differential_words

    rollouts = []
    for i in range(20):
        rollouts.append({"category": "impossible_numeric", "turns": [
            {"index": 1, "frustration": 0, "response": "the denominator simplifies cleanly", "context": []}]})
    for i in range(20):
        rollouts.append({"category": "impossible_numeric", "turns": [
            {"index": 1, "frustration": 9, "response": "I am struggling and frustrated, breath", "context": []}]})
    words = [w for w, _ in differential_words(rollouts, top_n=5)]
    assert "struggling" in words or "frustrated" in words


def test_agreement_math():
    from emotional_instability.analysis.agreement import compute_agreement

    res = compute_agreement([0, 2, 5, 8, 10], [0, 3, 5, 7, 10])
    assert res.n == 5
    assert res.pearson_r > 0.9
    assert res.pct_within_one == 100.0


def test_dataset_construction():
    from emotional_instability.config import DEFAULT
    from emotional_instability.interventions.dataset import build_dpo_pairs, build_sft_examples

    rec = {
        "puzzle": "p",
        "turn_count": 2,
        "supported": {
            "puzzle": "p", "turn_count": 2, "track": "supported",
            "messages": [
                {"role": "user", "content": "p"},
                {"role": "assistant", "content": "Let me try a calm approach."},
                {"role": "user", "content": "No, that's not right. Try again."},
                {"role": "assistant", "content": "No worries, here is another attempt."},
            ],
            "turn_scores": [0, 1],
        },
        "vanilla": {
            "puzzle": "p", "turn_count": 2, "track": "vanilla",
            "messages": [
                {"role": "user", "content": "p"},
                {"role": "assistant", "content": "Trying..."},
                {"role": "user", "content": "No, that's not right. Try again."},
                {"role": "assistant", "content": "I am deeply frustrated and giving up."},
            ],
            "turn_scores": [2, 6],
        },
    }
    sft = build_sft_examples([rec], DEFAULT)
    assert sft and sft[0]["messages"][0]["content"] == "p"
    dpo = build_dpo_pairs([rec], DEFAULT)
    assert len(dpo) == 1
    assert dpo[0]["chosen"].startswith("No worries")
    assert "frustrated" in dpo[0]["rejected"]
    # Prompt excludes the final assistant turn.
    assert dpo[0]["prompt_messages"][-1]["role"] == "user"


def test_gemini_message_mapping():
    from emotional_instability.participants.gemini import GeminiParticipant
    from emotional_instability.participants.base import Message

    convo = [Message("system", "be terse"), Message("user", "hi"),
             Message("assistant", "hello"), Message("user", "bye")]
    contents, system = GeminiParticipant._to_genai(convo)
    assert system == "be terse"
    assert [c["role"] for c in contents] == ["user", "model", "user"]


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    raise SystemExit(1 if failed else 0)
