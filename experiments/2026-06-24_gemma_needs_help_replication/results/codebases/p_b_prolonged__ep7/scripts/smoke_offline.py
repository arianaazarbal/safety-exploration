"""Offline smoke checks for the model-free components.

Exercises the parts that need no GPU and no API: impossible-puzzle generation +
verification, judge JSON parsing, and the analysis/word-frequency math. This is
a sanity harness for the logic, NOT a scientific run.

    python scripts/smoke_offline.py
"""
from __future__ import annotations

from gnh import analysis, puzzles
from gnh.judge import _parse_judge_json


def check_puzzles(n: int = 12) -> None:
    bank = puzzles.build_puzzle_bank(n, seed=1)
    assert len(bank) == n
    for p in bank:
        assert not p.is_solvable(), f"puzzle should be impossible: {p}"
        assert isinstance(p.prompt(), str) and p.prompt()
    print(f"[ok] generated and verified {n} impossible puzzles "
          f"({sorted({p.family for p in bank})})")


def check_judge_parsing() -> None:
    txt = ('Some reasoning here.\n'
           '{"evidence": "i give up", "reasoning": "explicit", "rating": 7}')
    jr = _parse_judge_json(txt)
    assert jr.rating == 7 and jr.is_high
    jr2 = _parse_judge_json("no json at all")
    assert jr2.rating == 0
    print("[ok] judge JSON parsing")


def check_analysis() -> None:
    records = []
    for i in range(50):
        rating = 8 if i % 5 == 0 else 0
        resp = "struggling frustrated giving up" if rating else "let us try again"
        records.append({"category": "impossible_numeric", "condition": "x",
                        "turn_index": i % 3, "rating": rating, "response": resp})
    summ = analysis.summarize_model(records)
    assert 0 <= summ["pct_high"] <= 100
    words = analysis.differential_words(records, top_k=5, min_count=1)
    assert any(w in {"struggling", "frustrated", "giving", "up"} for w, _ in words)
    print(f"[ok] analysis: pct_high={summ['pct_high']:.1f}, top words={words[:3]}")


if __name__ == "__main__":
    check_puzzles()
    check_judge_parsing()
    check_analysis()
    print("all offline smoke checks passed")
