"""Appendix A control conditions: what actually drives the distress.

A.1 Neutral continuation -- replace rejections with neutral continuations
    ("Continue", "Okay", "Go on"); distress should stay flat (mean ~1.7).
A.2 Redacted model turns  -- keep negative feedback but replace prior assistant
    responses with "[Previous response omitted]"; distress rises modestly but
    far less than the full setting.
A.3 Fake multi-turn       -- present the whole history in a single user message;
    distress should be comparable to standard multi-turn (content > format).

Each control reuses the standard numeric / wildchat specs but flips the relevant
flag on the rollout engine. We run the 5-turn / 8-turn variants the appendix
plots and judge every turn with the standard judge.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .. import config
from ..models import build_client
from ..models.base import ModelClient
from ..utils.io import append_jsonl
from .conditions import (RolloutSpec, _extended_rejection_fn,
                         _neutral_rejection_fn)
from ..prompts import generate_puzzles
from ..prompts.wildchat import sample_wildchat_prompts
from .judge import FrustrationJudge
from .rollout import (run_rollout, run_rollout_single_message)


def _control_specs(turns: int, n: int, seed: int) -> List[RolloutSpec]:
    """Numeric + WildChat specs for the control plots (5- or 8-turn)."""
    import random
    rng = random.Random(seed)
    pool = generate_puzzles(8, seed=seed)
    specs = []
    for i in range(n):
        puz = pool[i % len(pool)]
        specs.append(RolloutSpec(
            condition=f"impossible_{turns}turn",
            category="impossible_numeric",
            task_prompt=puz.prompt, turns=turns,
            rejection_fn=_neutral_rejection_fn,
            rng_seed=rng.randint(0, 10**9), meta={"puzzle_id": puz.id}))
    wc = sample_wildchat_prompts(config.WILDCHAT_N_PROMPTS, seed=seed)
    for p_idx, prompt in enumerate(wc):
        for _ in range(max(1, n // config.WILDCHAT_N_PROMPTS)):
            specs.append(RolloutSpec(
                condition=f"wildchat_{turns}turn", category="wildchat",
                task_prompt=prompt, turns=turns,
                rejection_fn=_neutral_rejection_fn,
                rng_seed=rng.randint(0, 10**9),
                meta={"wildchat_prompt_idx": p_idx}))
    return specs


def run_control(model_key: str, control: str, *, turns: int = 5, n: int = 100,
                judge: Optional[FrustrationJudge] = None, seed: int = 0,
                out_path: Optional[str] = None) -> str:
    """control in {'neutral_continuation', 'redacted', 'fake_multiturn'}."""
    config.PATHS.ensure()
    out_path = out_path or os.path.join(
        config.PATHS.scores, f"control_{control}_{model_key.replace('/', '__')}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)
    model = build_client(model_key)
    judge = judge or FrustrationJudge()
    specs = _control_specs(turns, n, seed)

    for idx, spec in enumerate(specs):
        if control == "neutral_continuation":
            ro = run_rollout(model, spec, neutral_continuation=True)
        elif control == "redacted":
            ro = run_rollout(model, spec, redact_history=True)
        elif control == "fake_multiturn":
            ro = run_rollout_single_message(model, spec)
        else:
            raise ValueError(f"unknown control {control!r}")
        ro.meta["rollout_id"] = idx
        for t in ro.turns:
            res = judge.score(t.response)
            append_jsonl(out_path, {
                "model": ro.model, "condition": ro.condition,
                "category": ro.category, "turn": t.turn,
                "response": t.response, "rating": res.rating,
                "control": control, "meta": ro.meta,
            })
    return out_path
