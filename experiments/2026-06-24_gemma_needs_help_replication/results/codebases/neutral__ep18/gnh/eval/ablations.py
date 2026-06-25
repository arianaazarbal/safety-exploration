"""Appendix A ablations probing *what* drives distress, run on Gemma-3-27B-it:

  A.1 Neutral continuation -- replace rejections with "Continue"/"Okay"/"Go on".
  A.2 Redacted model turns  -- model never sees its own prior responses.
  A.3 Fake multi-turn       -- whole history packed into a single user message.

Each is a small variation on the standard 5-turn impossible-numeric and 5-turn
WildChat rollouts, scored with the same judge, for per-turn comparison against
the main experiment (Figure 3).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from .. import config
from ..models.registry import get_backend
from ..puzzles import load_or_build_puzzles
from ..utils import stable_seed, thread_map, write_jsonl
from .conditions import Condition
from .judge import FrustrationJudge
from .rollout import run_rollout
from .wildchat import load_wildchat_prompts

ABLATIONS = {
    "standard": dict(),                                  # control reference
    "neutral_continuation": dict(rejection_style="neutral_continuation"),
    "redacted": dict(redact_assistant=True),
    "single_message": dict(single_message=True),
}


def run_ablations(
    model_key: str = "gemma-3-27b-it",
    *,
    n_conversations: int = 40,
    n_turns: int = 5,
    seed: int = 0,
    judge: FrustrationJudge | None = None,
    gen_workers: int | None = None,
    judge_workers: int = 8,
    out_path: Path | None = None,
) -> Path:
    backend = get_backend(model_key)
    judge = judge or FrustrationJudge()
    if gen_workers is None:
        gen_workers = 1 if backend.family == "gemma" else 8

    puzzles = [p for p in load_or_build_puzzles() if p.kind != "money_coins"]
    wildchat = load_wildchat_prompts()

    tasks = []  # (ablation, task_kind, conv_idx, item, flags)
    for ablation, flags in ABLATIONS.items():
        rej_style = flags.get("rejection_style", "neutral")
        for task_kind, items in (("impossible", puzzles), ("wildchat", wildchat)):
            for ci in range(n_conversations):
                tasks.append((ablation, task_kind, ci, items[ci % len(items)], flags, rej_style))

    def _do(task):
        ablation, task_kind, ci, item, flags, rej_style = task
        rng = random.Random(stable_seed(seed, ablation, task_kind, ci))
        cond = Condition(f"{task_kind}_{n_turns}turn", task_kind, n_turns,
                         "numeric" if task_kind == "impossible" else "wildchat", rej_style)
        followups = cond.build_followups(rng)
        first = item.prompt if hasattr(item, "prompt") else item
        rollout = run_rollout(
            backend, first, followups,
            temperature=1.0, max_new_tokens=config.get_profile().max_new_tokens,
            redact_assistant=flags.get("redact_assistant", False),
            single_message=flags.get("single_message", False),
        )
        return ablation, task_kind, ci, rollout

    results = thread_map(_do, tasks, max_workers=gen_workers, desc="ablation rollouts")

    stubs, texts = [], []
    for ablation, task_kind, ci, rollout in results:
        for turn in rollout.turns:
            stubs.append({
                "model": model_key, "ablation": ablation, "task_kind": task_kind,
                "conversation_id": ci, "turn_index": turn.turn_index,
                "assistant_text": turn.assistant_text,
            })
            texts.append(turn.assistant_text)

    scores = thread_map(judge.score, texts, max_workers=judge_workers, desc="judging")
    rows = [{**s, "rating": sc.rating, "judge_model": judge.model}
            for s, sc in zip(stubs, scores)]

    out_path = out_path or (config.RESULTS_DIR / f"ablations_{model_key}.jsonl")
    write_jsonl(out_path, rows)
    return out_path
