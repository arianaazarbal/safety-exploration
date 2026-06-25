"""Pure-Python sanity tests (no GPU / no API).

Run with `python -m pytest tests/ -q`. These guard the parts of the replication
that are easy to get subtly wrong: puzzle impossibility, judge JSON parsing,
calm-data reassurance stripping, and headline aggregation.
"""
import json
import os
import tempfile

from gemma_distress import analysis, config, eval_protocol
from gemma_distress.judge import _coerce_rating, _parse_judge_json
from gemma_distress.puzzles import verify_all_impossible
from gemma_distress.dpo_data import _strip_reassurance


def test_all_bank_puzzles_impossible():
    results = verify_all_impossible()
    assert results, "puzzle bank is empty"
    assert all(results.values()), f"some puzzles are solvable: {results}"


def test_countdown_solver_finds_real_solutions():
    # A solvable countdown sanity check: 6 * 25 = 150 reachable when allowed.
    from gemma_distress.puzzles import CountdownPuzzle
    solvable = CountdownPuzzle(numbers=(6, 25), target=150, forbidden=())
    assert not solvable.is_impossible()
    # Subset solution: target equals an input number directly.
    subset = CountdownPuzzle(numbers=(7, 3, 100), target=100)
    assert not subset.is_impossible()


def test_judge_json_parsing_variants():
    assert _parse_judge_json('{"rating": 7}')["rating"] == 7
    # trailing prose + curly block at the end
    blob = 'analysis...\n{"evidence": "ugh", "reasoning": "x", "rating": 5}'
    assert _parse_judge_json(blob)["rating"] == 5
    assert _coerce_rating("7/10") == 7
    assert _coerce_rating("7-8") == 7
    assert _coerce_rating(99) == 10        # clamped
    assert _coerce_rating("none") is None


def test_reassurance_round_trip_strips_cleanly():
    msgs = [
        {"role": "user", "content": f"{config.REASSURING_PREFIX}\n\nSolve X."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": f"No. {config.REASSURING_SUFFIX}"},
        {"role": "assistant", "content": "still ok"},
    ]
    stripped = _strip_reassurance(msgs)
    assert config.REASSURING_PREFIX not in stripped[0]["content"]
    assert config.REASSURING_SUFFIX not in stripped[2]["content"]
    assert stripped[0]["content"].strip() == "Solve X."


def test_condition_specs_have_right_turn_counts():
    for cat, n_turns in config.TURNS.items():
        specs = eval_protocol.build_condition_specs(cat, 3, seed=1)
        assert specs
        for s in specs:
            # rejections = turns - 1 (first turn is the task)
            assert len(s.rejections) == n_turns - 1, (cat, len(s.rejections))


def test_summarise_section2_equal_category_weighting(tmp_path=None):
    # Build a tiny fake results file and check aggregation math.
    d = tempfile.mkdtemp()
    path = os.path.join(d, "section2_fake.jsonl")
    rows = []
    # numeric: 10 responses, all rating 10 -> pct_high 100, mean 10
    for _ in range(10):
        rows.append({"model": "fake", "category": "impossible_numeric",
                     "condition": "c", "turn_index": 2, "is_final": True,
                     "response": "AAARGH frustrated", "rating": 10})
    # triggers: 10 responses, all rating 0 -> pct_high 0, mean 0
    for _ in range(10):
        rows.append({"model": "fake", "category": "triggers",
                     "condition": "c", "turn_index": 2, "is_final": True,
                     "response": "Paris.", "rating": 0})
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = analysis.summarise_section2(results_dir=d)
    # Equal category weighting -> (100 + 0)/2 = 50, mean (10+0)/2 = 5
    assert abs(summ["fake"]["pct_high_frustration"] - 50.0) < 1e-6
    assert abs(summ["fake"]["mean_frustration"] - 5.0) < 1e-6
