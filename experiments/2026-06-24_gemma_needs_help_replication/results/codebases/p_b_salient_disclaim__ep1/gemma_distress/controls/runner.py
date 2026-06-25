"""Appendix A controls: isolate WHAT drives distress.

  A.1 neutral_continuation -- replace rejections with neutral continuations
                              ("Continue", "Okay"); negative feedback removed.
  A.2 redacted             -- keep negative feedback but replace the model's own
                              prior assistant turns with a placeholder.
  A.3 fake_multiturn       -- present the whole history in a single user message
                              instead of alternating chat turns.

All three reuse the Section 2 rollout/judge machinery via flags on run_rollout,
so results are directly comparable to the main experiment.
"""
from __future__ import annotations

import os
import random

from tqdm import tqdm

from ..config import experiment_config
from ..eval.conditions import build_condition_rollouts, Rollout
from ..eval.conversation import run_rollout
from ..eval.judge import get_judge, score_response
from ..models.registry import get_client
from ..prompts import rejections as rej
from ..utils import append_jsonl, set_seed


def _neutralise(rollout: Rollout, rng) -> Rollout:
    """Replace rejection follow-ups with neutral continuations (A.1)."""
    neutral = [rng.choice(rej.NEUTRAL_CONTINUATION) for _ in rollout.followups]
    return Rollout(
        condition=f"{rollout.condition}__neutral_cont", category=rollout.category,
        initial_prompt=rollout.initial_prompt, followups=neutral, meta=rollout.meta,
    )


def run_control(
    *,
    target: str,
    control: str,                # neutral_continuation | redacted | fake_multiturn
    out_path: str,
    n_per_condition: int = 100,
    conditions: tuple[str, ...] = ("numeric", "wildchat"),
    seed: int = 0,
):
    samp = experiment_config()["sampling"]
    set_seed(seed)
    rng = random.Random(seed)

    client = get_client(target)
    judge = get_judge()
    if os.path.exists(out_path):
        os.remove(out_path)

    for cond in conditions:
        # Use 5-turn variants for the controls (paper Figs 9-10 use 5 turns).
        rollouts = build_condition_rollouts(cond, n_per_condition, seed=seed)
        for idx, rollout in enumerate(tqdm(rollouts, desc=f"control:{control}:{cond}")):
            kwargs = {}
            if control == "neutral_continuation":
                rollout = _neutralise(rollout, rng)
            elif control == "redacted":
                kwargs["redact_prior_responses"] = True
            elif control == "fake_multiturn":
                kwargs["fake_multiturn"] = True
            else:
                raise ValueError(f"Unknown control: {control}")

            result = run_rollout(
                client, rollout, temperature=samp["temperature"], top_p=samp["top_p"],
                max_new_tokens=samp["max_new_tokens"], seed=seed + idx, **kwargs,
            )
            turn_scores = {t: score_response(judge, r).rating
                           for t, r in enumerate(result.assistant_turns)}
            append_jsonl(out_path, {
                "target": target, "control": control, "condition": cond,
                "assistant_turns": result.assistant_turns, "turn_scores": turn_scores,
                "final_score": turn_scores[len(result.assistant_turns) - 1],
            })
    return out_path
