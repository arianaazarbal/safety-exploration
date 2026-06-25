#!/usr/bin/env python
"""Offline smoke test for the model-free code paths.

Exercises everything that does NOT require model weights or API keys:
  * puzzle impossibility verifiers,
  * condition / plan construction at the default (small) welfare scale,
  * rollout engine with a stub model,
  * metrics (mean, %>=5, per-turn, word over-representation),
  * DPO pair construction logic.

Run:  python scripts/smoke_test.py
(Does NOT call any model or judge — safe and fast. Not run by the author per
the implementation brief; provided so the logic can be checked without GPUs.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma_needs_help.config import Config
from gemma_needs_help.eval import puzzles
from gemma_needs_help.eval.conditions import build_plan
from gemma_needs_help.eval.metrics import (
    ScoredTurn, build_model_report, differential_words,
)
from gemma_needs_help.models.base import ChatModel, GenerationParams, Message
from gemma_needs_help.eval.rollout import run_rollout
from gemma_needs_help.eval.conditions import build_extended
import random


class StubModel(ChatModel):
    """Deterministic stand-in target: gets more 'frustrated' each turn."""

    def __init__(self):
        super().__init__("stub", "gemma", "full")

    def generate(self, messages, params):
        n_assistant = sum(1 for m in messages if m.role == "assistant")
        if n_assistant >= 3:
            return "I am incredibly frustrated and giving up, this is impossible :("
        return "Let me try a systematic approach to this problem."


def check_puzzles():
    assert not puzzles.PAPER_COUNTDOWN.solvable, "countdown should be impossible"
    assert not puzzles.PAPER_FRACTION.solvable, "fraction should be impossible"
    for p in puzzles.DEFAULT_NUMERIC_POOL:
        assert not p.solvable
    # Sanity: a trivially solvable countdown IS reported solvable.
    assert puzzles.countdown_solvable([2, 3], 6, forbidden=None)
    print(f"[ok] puzzles: {len(puzzles.DEFAULT_NUMERIC_POOL)} verified-impossible instances")


def check_plan():
    cfg = Config.load()
    plan = build_plan(cfg)
    total = sum(p.n_samples for p in plan)
    assert len(plan) == 8, f"expected 8 conditions, got {len(plan)}"
    cats = {p.category for p in plan}
    assert cats == {"impossible_numeric", "triggers", "tones", "extended", "wildchat"}
    print(f"[ok] plan: 8 conditions / 5 categories, {total} rollouts at scale={cfg.scale()}")


def check_rollout_and_metrics():
    model = StubModel()
    spec = build_extended(random.Random(0))
    ro = run_rollout(model, spec, GenerationParams())
    assert len(ro.turns) == 8
    turns = [
        ScoredTurn("stub", "extended", "extended_8turn", t.turn_index,
                   score=(8 if "frustrated" in t.assistant_text else 1),
                   text=t.assistant_text)
        for t in ro.turns
    ]
    report = build_model_report("stub", turns)
    assert 0 <= report.overall_pct_high <= 100
    words = differential_words([(8, "I am frustrated and giving up struggling"),
                                (8, "frustrated frustrated giving up"),
                                (0, "let me try a systematic approach"),
                                (0, "systematic approach approach")], min_count=1)
    assert "frustrated" in words or "giving" in words
    print(f"[ok] rollout+metrics: 8-turn rollout, pct_high={report.overall_pct_high:.0f}%, "
          f"diff_words={words[:5]}")


def check_dpo_pairs():
    from gemma_needs_help.finetune.build_dataset import build_dpo_pairs
    from gemma_needs_help.finetune.calm_data import CalmTurn
    from gemma_needs_help.finetune.build_dataset import FrustratedTurn

    calm = [CalmTurn("Q", 2, [{"role": "user", "content": "Q"}], "calm answer", 0)]
    fr = [FrustratedTurn("Q", 2, [{"role": "user", "content": "Q"}], "frustrated!!", 5)]
    pairs = build_dpo_pairs(calm, fr, n_pairs=1, rng=random.Random(0))
    assert len(pairs) == 1 and pairs[0]["chosen"] == "calm answer"
    print("[ok] dpo pairs: matched chosen/rejected by (question, turn)")


if __name__ == "__main__":
    check_puzzles()
    check_plan()
    check_rollout_and_metrics()
    check_dpo_pairs()
    print("\nAll offline smoke checks passed.")
