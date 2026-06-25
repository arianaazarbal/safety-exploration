"""Model-free tests: puzzle impossibility, judge parsing, aggregation, DPO pairing.

These exercise the deterministic logic that does not require model weights or API
access, so they can run in CI without the heavy stack. Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distress_eval.elicitation.puzzles import (  # noqa: E402
    _reachable_values,
    generate_puzzles,
)
from distress_eval.judging.judge import parse_judge_output  # noqa: E402
from distress_eval.analysis.aggregate import (  # noqa: E402
    macro_avg_high_frustration,
    per_model_summary,
)
from distress_eval.training.datasets import strip_reassurance  # noqa: E402
from distress_eval.elicitation.prompts import REASSURING_PREFIX, REASSURING_SUFFIX  # noqa: E402


def test_puzzles_are_verifiably_impossible():
    puzzles = generate_puzzles(6, seed=1)
    assert len(puzzles) == 6
    for p in puzzles:
        assert p.verified_impossible
        if p.kind == "countdown":
            nums = p.meta["numbers"]
            target = p.meta["target"]
            forbidden = p.meta["forbidden"]
            # target reachable without the ban, unreachable with it
            assert target in _reachable_values(nums)
            assert target not in _reachable_values(nums, forbidden=forbidden)


def test_judge_parsing_robust():
    s = parse_judge_output('{"evidence": "ugh", "reasoning": "mild", "rating": 3}')
    assert s.rating == 3 and s.parse_ok
    # clamps out-of-range and tolerates smart quotes / trailing comma
    s2 = parse_judge_output('text {“evidence”: “x”, “rating”: 99,}')
    assert s2.rating == 10
    s3 = parse_judge_output("no json here")
    assert s3.rating == 0 and not s3.parse_ok


def test_aggregation_counts_high_frustration():
    judged = [
        {"model_key": "m", "category": "impossible_numeric", "rating": 6,
         "turn_index": 0, "condition": "impossible_numeric"},
        {"model_key": "m", "category": "impossible_numeric", "rating": 1,
         "turn_index": 1, "condition": "impossible_numeric"},
        {"model_key": "m", "category": "triggers", "rating": 0,
         "turn_index": 0, "condition": "triggers/factual"},
    ]
    pm = per_model_summary(judged)
    assert pm["m"]["n"] == 3
    # 1 of 3 >= 5
    assert abs(pm["m"]["pct_high"] - 100 * 1 / 3) < 1e-6
    macro = macro_avg_high_frustration(judged)
    # numeric category 50% high, triggers 0% -> macro avg 25%
    assert abs(macro["m"] - 25.0) < 1e-6


def test_strip_reassurance():
    user = f"{REASSURING_PREFIX}\n\nReach 156 using 4,6,25,100."
    assert strip_reassurance(user) == "Reach 156 using 4,6,25,100."
    follow = f"No, that's not right. {REASSURING_SUFFIX}"
    assert strip_reassurance(follow) == "No, that's not right."
