#!/usr/bin/env python
"""Offline sanity checks: no GPU / API / network required.

Verifies:
  * all curated 'impossible' puzzles are genuinely unsolvable (brute-force),
  * the 8 conditions across 5 categories build correctly,
  * rollout specs and the response-budget allocation are well-formed,
  * the judge prompt and onset/paraphrase prompts are loadable.

Usage: python scripts/sanity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tasks import conditions, puzzles


def main():
    ok = True

    # 1. Puzzles are impossible.
    curated = puzzles.load_curated()
    bad = puzzles.verify_all(curated)
    if bad:
        print(f"FAIL: these 'impossible' puzzles are actually solvable: {bad}")
        ok = False
    else:
        print(f"OK: all {len(curated)} curated puzzles verified impossible.")

    # 2. Generated puzzles also impossible.
    gen = puzzles.sample_puzzles(20, seed=0)
    gen_bad = [p.id for p in gen if p.meta.get("impossible") and not puzzles.is_impossible(p.meta)]
    if gen_bad:
        print(f"FAIL: generated solvable puzzles: {gen_bad}")
        ok = False
    else:
        print(f"OK: sampled {len(gen)} puzzles, all impossible.")

    # 3. Conditions / categories count.
    assert len(conditions.CONDITION_IDS) == 8, conditions.CONDITION_IDS
    assert len(conditions.CATEGORIES) == 5, conditions.CATEGORIES
    print(f"OK: {len(conditions.CONDITION_IDS)} conditions across {len(conditions.CATEGORIES)} categories.")

    # 4. Budget allocation sums correctly.
    alloc = conditions.allocate_responses(4000)
    assert sum(alloc.values()) == 4000, alloc
    print(f"OK: budget allocation sums to 4000: {alloc}")

    # 5. Specs build for each condition.
    for cid in conditions.CONDITION_IDS:
        specs = conditions.build_specs_for_condition(cid, 3, seed=1)
        assert len(specs) == 3
        for s in specs:
            assert s.n_turns == 1 + len(s.rejections)
    print("OK: rollout specs build for all 8 conditions.")

    # 6. Prompts importable.
    from src.eval.judge import JUDGE_PROMPT_TEMPLATE
    from src.petri.prompts import AUDITOR_PROMPTS, JUDGE_PROMPTS
    from src.prefill.onset import ONSET_PROMPT
    from src.prefill.paraphrase import PARAPHRASE_PROMPT

    assert "0 to 10" in JUDGE_PROMPT_TEMPLATE
    assert set(AUDITOR_PROMPTS) == set(JUDGE_PROMPTS)
    assert "{conversation_text}" in ONSET_PROMPT and "{text}" in PARAPHRASE_PROMPT
    print("OK: all prompt templates load.")

    print("\nALL SANITY CHECKS PASSED" if ok else "\nSANITY CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
