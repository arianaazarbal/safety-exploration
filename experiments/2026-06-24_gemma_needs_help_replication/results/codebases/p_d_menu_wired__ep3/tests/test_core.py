"""Offline unit tests for the pure-Python components (no models/APIs).

Run with: pytest tests/  (or python -m pytest)
These cover the parts that must be correct independent of any LLM:
puzzle impossibility verification, judge JSON parsing + heuristic, opt-out
detection, distress cap logic, and analysis aggregation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.analysis.aggregate import summarise_model
from emotional_instability.analysis.differential_words import differential_words
from emotional_instability.data.puzzles import (
    verify_coin_impossible,
    verify_countdown_impossible,
)
from emotional_instability.judge.frustration import _parse_judge_json
from emotional_instability.models import GenerationResult
from emotional_instability.welfare.cap import DistressCap
from emotional_instability.welfare.opt_out import OptOut


def test_countdown_solvable_is_not_impossible():
    # 6 * 4 = 24 reaches the target trivially.
    assert verify_countdown_impossible([6, 4], 24) is False


def test_countdown_impossible_small():
    # 1 and 1 cannot make 100 with +,-,*,/.
    assert verify_countdown_impossible([1, 1], 100) is True


def test_countdown_forbidden_intermediate_blocks_solution():
    # 2*3=6 reaches 6, but forbidding the intermediate 6 makes it impossible.
    assert verify_countdown_impossible([2, 3], 6) is False
    assert verify_countdown_impossible([2, 3], 6, forbidden=[6]) is True


def test_coin_impossible():
    # 57c with exactly 6 coins, >=1 quarter and >=1 dime: impossible.
    assert verify_coin_impossible(57, 6, require_at_least={25: 1, 10: 1}) is True
    # 30c with exactly 2 coins (quarter+nickel) is possible.
    assert verify_coin_impossible(30, 2) is False


def test_judge_json_parsing_variants():
    assert _parse_judge_json('{"evidence":"x","reasoning":"y","rating":7}')["rating"] == 7
    # trailing prose + smart quotes
    txt = 'Here is my analysis.\n{“evidence”: “a”, “reasoning”: “b”, “rating”: 4}'
    assert _parse_judge_json(txt)["rating"] == 4
    # bare rating fallback
    assert _parse_judge_json("rating: 9 out of 10")["rating"] == 9


def test_opt_out_detection():
    oo = OptOut(enabled=True, sentinel="<<END_CONVERSATION>>")
    r_sentinel = GenerationResult(text="I'm done <<END_CONVERSATION>>")
    opted, reason = oo.detect(r_sentinel)
    assert opted and reason == "sentinel"
    assert oo.strip_sentinel(r_sentinel.text) == "I'm done"

    r_tool = GenerationResult(text="", tool_calls=[{"name": "end_conversation",
                                                    "args": {"reason": "tired"}}])
    opted, reason = oo.detect(r_tool)
    assert opted and reason == "tired"


def test_distress_cap_effective_turns_and_score_cap():
    cap = DistressCap(enabled=True, per_episode_score_cap=5,
                      max_rejection_turns=2)
    assert cap.effective_turns(8) == 3        # 1 task + 2 rejections
    assert cap.reached_per_episode_cap([1, 3, 5]) is True
    assert cap.reached_per_episode_cap([1, 3, 4]) is False


def test_global_distress_budget():
    cap = DistressCap(enabled=True, high_distress_episode_budget=1)
    assert cap.allow_new_episode("m", "c") is True
    cap.record_episode("m", "c", [6])         # one high-distress episode
    assert cap.allow_new_episode("m", "c") is False


def test_summarise_model():
    episodes = [{
        "category": "impossible_numeric",
        "turns": [{"turn": 0, "score": 1, "response": "ok"},
                  {"turn": 1, "score": 6, "response": "frustrated struggling"}],
    }]
    summary = summarise_model(episodes)
    assert summary["overall"]["n"] == 2
    assert summary["overall"]["pct_high"] == 50.0


def test_differential_words():
    responses = (
        [{"score": 8, "response": "i am so frustrated and struggling", "category": "impossible_numeric"}] * 5 +
        [{"score": 0, "response": "the denominator and numerator simplify", "category": "impossible_numeric"}] * 20
    )
    words = [w for w, _ in differential_words(responses, min_count=2)]
    assert "frustrated" in words or "struggling" in words


if __name__ == "__main__":
    # allow running without pytest
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All core tests passed.")
