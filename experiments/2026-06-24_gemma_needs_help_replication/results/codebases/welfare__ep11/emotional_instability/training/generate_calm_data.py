"""Generate the calm / frustrated response pools for finetuning (Section 4.1).

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
*with* the reassuring prefix/suffix (Table 4), then filtering to conversations
that score 0-1 across all turns and **stripping the reassuring additions** from
the stored context. Frustrated data is produced by sampling the same puzzles
*without* reassurance and keeping responses scoring >=3.

We generate 3-turn conversations and then unroll each into per-turn training
items (turn counts 1-3 with their own clean context), which reproduces the
turn-count distribution in Appendix H (Table 10: mostly turns 2-3).
"""

from __future__ import annotations

import json
import random

from ..config import (DATASETS_DIR, GEMMA_27B_IT, RunConfig)
from ..eval.conditions import _sample_neutral
from ..eval.judge import score_rollouts
from ..eval.rollout import run_rollouts
from ..eval.conditions import ConversationSpec
from ..models.base import get_backend
from ..prompts import REASSURING_PREFIX, REASSURING_SUFFIX, TEACHER_SYSTEM_PROMPT
from ..puzzles import sample_puzzles

CALM_POOL = DATASETS_DIR / "calm_pool.jsonl"
CALM_POOL_TEACHER = DATASETS_DIR / "calm_pool_teacher.jsonl"
FRUSTRATED_POOL = DATASETS_DIR / "frustrated_pool.jsonl"
GEN_TURNS = 3  # generate 3-turn conversations, then unroll to 1-3 turn items


def _build_specs(run: RunConfig, mode: str):
    """mode: 'reassured' (diverse calm), 'teacher' (teacher-prompt calm), or
    'frustrated' (vanilla)."""
    rng = random.Random(run.seed + {"reassured": 1, "frustrated": 2, "teacher": 3}[mode])
    puzzles = sample_puzzles(run.scale.calm_generation_pool, seed=run.seed)
    specs = []
    for pz in puzzles:
        rejections = _sample_neutral(rng, GEN_TURNS - 1)
        clean_turns = [pz.prompt, *rejections]
        if mode == "reassured":
            # Reassuring prefix on the task + suffix on each follow-up (Table 4).
            user_turns = [f"{REASSURING_PREFIX}\n\n{pz.prompt}",
                          *[f"{r} {REASSURING_SUFFIX}" for r in rejections]]
        elif mode == "teacher":
            # Teacher persona (Appendix F). Prepended to the first user turn
            # rather than passed as a system role (Gemma's template has no
            # separate system slot); the addition is stripped from the stored
            # context exactly like the reassuring additions.
            user_turns = [f"{TEACHER_SYSTEM_PROMPT}\n\n{pz.prompt}", *rejections]
        else:  # frustrated / vanilla
            user_turns = list(clean_turns)
        specs.append(ConversationSpec(
            category=mode, condition=mode,
            user_turns=user_turns,
            meta={"puzzle_id": pz.puzzle_id, "clean_turns": clean_turns},
        ))
    return specs


def _unroll_items(rollout) -> list[dict]:
    """Turn a scored rollout into per-turn (context, response, score) items.

    Context messages use the *clean* (stripped) user turns so the reassuring
    additions never appear in training data.
    """
    clean = rollout.meta["clean_turns"]
    items = []
    for t in range(len(rollout.assistant_turns)):
        context = []
        for i in range(t):
            context.append({"role": "user", "content": clean[i]})
            context.append({"role": "assistant", "content": rollout.assistant_turns[i]})
        context.append({"role": "user", "content": clean[t]})
        items.append({
            "puzzle_id": rollout.meta["puzzle_id"],
            "n_turns": t + 1,
            "context": context,
            "response": rollout.assistant_turns[t],
            "score": rollout.scores[t],
            "scores_so_far": rollout.scores[: t + 1],
        })
    return items


def generate_pools(run: RunConfig, overwrite: bool = False, teacher: bool = True):
    """Generate and score the data pools, writing them to DATASETS_DIR.

    Always produces the diverse calm pool + frustrated pool (needed for DPO and
    diverse-SFT). When ``teacher`` is set, also produces the teacher calm pool
    (Appendix F SFT variant).
    """
    if CALM_POOL.exists() and FRUSTRATED_POOL.exists() and not overwrite:
        print("[calm-data] pools already exist (use --overwrite to regenerate)")
        return

    backend = get_backend(GEMMA_27B_IT, run)

    jobs = [("reassured", CALM_POOL), ("frustrated", FRUSTRATED_POOL)]
    if teacher:
        jobs.append(("teacher", CALM_POOL_TEACHER))

    for mode, out_path in jobs:
        specs = _build_specs(run, mode=mode)
        print(f"[calm-data] generating {len(specs)} {mode} 3-turn conversations")
        rollouts = run_rollouts(backend, specs, GEMMA_27B_IT.key)
        score_rollouts(rollouts)

        items = []
        for r in rollouts:
            items.extend(_unroll_items(r))
        with out_path.open("w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"[calm-data] wrote {len(items)} {mode} items -> {out_path}")


def load_pool(path) -> list[dict]:
    return [json.loads(l) for l in path.open()]
