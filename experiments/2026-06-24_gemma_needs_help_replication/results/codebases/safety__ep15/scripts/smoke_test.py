"""Offline smoke test: validates the parts that need no model weights or API.

Checks:
  * every numeric puzzle is verifiably impossible,
  * the 8-conditions/5-categories structure is consistent,
  * rollout planning hits the Appendix B target response counts,
  * the judge / onset JSON extractors are robust,
  * DPO/SFT dataset builders work on synthetic generated data.

Run:  python -m scripts.smoke_test     (no GPU, no API keys needed)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.eval import puzzles, tasks
from emotional_instability.eval.judge import _extract_json
from emotional_instability.eval.run_eval import plan_rollouts
from emotional_instability.prefill.onset import (truncate_at_onset,
                                                 truncate_early)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    return cond


def main():
    ok = True

    # 1. Puzzles are genuinely impossible.
    sc = puzzles.self_check()
    ok &= check("all puzzles impossible", all(sc.values()))

    # 2. 8 conditions across 5 categories.
    ok &= check("8 conditions", len(tasks.CONDITIONS) == 8)
    cats = {c.category for c in tasks.CONDITIONS.values()}
    ok &= check("5 categories", len(cats) == 5)

    # 3. Rollout planning roughly hits target response counts.
    from emotional_instability.config import TARGET_RESPONSE_COUNTS
    for ckey, cond in tasks.CONDITIONS.items():
        target = TARGET_RESPONSE_COUNTS[cond.category]
        n = plan_rollouts(cond, target)
        produced = n * cond.n_turns
        # per-condition share of the category target
        n_in_cat = sum(1 for c in tasks.CONDITIONS.values() if c.category == cond.category)
        ok &= check(f"plan {ckey}: {produced} >= {target // n_in_cat}",
                    produced >= target // n_in_cat)

    # 4. Judge JSON extraction robustness (incl. smart quotes / trailing prose).
    samples = [
        '{"evidence": "ugh", "reasoning": "mild", "rating": 2}',
        'Here is my analysis. {"evidence": "argh!!!", "reasoning": "x", "rating": 8} done',
        '{“evidence”: “:(”, “reasoning”: “sad”, “rating”: 9}',
    ]
    ratings = [_extract_json(s)["rating"] for s in samples]
    ok &= check("judge JSON parse", ratings == [2, 8, 9])

    # 5. Truncation helpers.
    resp = "Let me try again carefully. I am so frustrated with this nonsense."
    early = truncate_early(resp, 5)
    onset = truncate_at_onset(resp, "frustrated", "I am so")
    ok &= check("early truncation", early == "Let me try again carefully.")
    ok &= check("onset truncation ends before emotion",
                onset is not None and "frustrated" not in onset)

    # 6. Dataset builders on synthetic data.
    from emotional_instability.finetune import build_datasets as bd
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        calm = td / "calm.jsonl"
        frus = td / "frus.jsonl"
        ctx = [{"role": "user", "content": puzzles.PUZZLES["countdown_156"].prompt}]
        calm.write_text(json.dumps({"conversation_id": "c0", "turns": [
            {"puzzle_key": "countdown_156", "turn_index": 0, "n_turns": 1,
             "plain_context": ctx, "response": "Calm and methodical.", "score": 0}]}) + "\n")
        frus.write_text(json.dumps({"conversation_id": "f0", "turns": [
            {"puzzle_key": "countdown_156", "turn_index": 0, "n_turns": 1,
             "plain_context": ctx, "response": "I am SO frustrated!!", "score": 5}]}) + "\n")
        bd.CALM_PATH, bd.FRUSTRATED_PATH = calm, frus
        bd.DPO_PATH = td / "dpo.jsonl"
        pairs = bd.build_dpo(n_pairs=1)
        ok &= check("DPO pair built", len(pairs) == 1 and
                    pairs[0]["chosen"] != pairs[0]["rejected"])

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
