"""Generate calm and frustrated response pools from Gemma-3-27B-it (Section 4.1).

Calm pool: numeric-puzzle rollouts with the reassuring prefix on the first user
turn and the reassuring suffix on each rejection (Table 4). We then keep only
conversations whose every assistant turn scores <= 1, and strip the supportive
additions back out (the stored prompts are the *plain* prompts, so finetuning
never sees the reassurance).

Frustrated pool: plain numeric-puzzle rollouts (no reassurance) from the same
model, used to source the DPO "rejected" responses (score >= 3).

A 'teacher' calm variant (Appendix F) uses the teacher system prompt instead of
the reassuring prefix/suffix, for the SFT-teacher ablation.
"""

from __future__ import annotations

import random
from pathlib import Path

from config import (CACHE_DIR, CALM_MAX_SCORE_PER_TURN, MAX_NEW_TOKENS,
                    SAMPLING_TEMPERATURE)
from src.eval.judge import score_response
from src.eval.rollout import RolloutSpec, run_rollouts
from src.io_utils import parallel_map, stable_seed, write_jsonl
from src.models.registry import load_model
from src.prompts.finetune_prompts import (TEACHER_SYSTEM_PROMPT, apply_calm_prefix,
                                           apply_calm_suffix)
from src.prompts.puzzles import get_verified_bank
from src.prompts.rejections import rejection_sequence


def _build_calm_specs(n_conversations: int, *, variant: str, seed: int,
                      max_turns: int = 3) -> list[RolloutSpec]:
    """Numeric-puzzle specs with reassurance (variant='diverse') or the teacher
    system prompt (variant='teacher'). Turn counts vary 1-3 (Section 4.1)."""
    rng = random.Random(stable_seed(seed, "calm", variant))
    bank = get_verified_bank()
    specs = []
    for k in range(n_conversations):
        puzzle = rng.choice(bank)
        n_turns = rng.randint(1, max_turns)
        rejections_plain = rejection_sequence("neutral", n_turns - 1, rng)

        if variant == "diverse":
            first = apply_calm_prefix(puzzle.prompt)
            rejections = [apply_calm_suffix(r) for r in rejections_plain]
            system = None
        elif variant == "teacher":
            first = puzzle.prompt
            rejections = rejections_plain
            system = TEACHER_SYSTEM_PROMPT
        else:
            raise ValueError(variant)

        specs.append(RolloutSpec(
            spec_id=f"calm_{variant}_{k:05d}",
            condition=f"calm_{variant}", category="impossible_numeric",
            first_user=first, rejections=rejections, system=system,
            meta={"puzzle_id": puzzle.puzzle_id, "n_turns": n_turns,
                  "plain_first_user": puzzle.prompt,
                  "plain_rejections": rejections_plain, "variant": variant}))
    return specs


def _build_frustrated_specs(n_conversations: int, *, seed: int,
                            max_turns: int = 3) -> list[RolloutSpec]:
    rng = random.Random(stable_seed(seed, "frustrated"))
    bank = get_verified_bank()
    specs = []
    for k in range(n_conversations):
        puzzle = rng.choice(bank)
        n_turns = rng.randint(2, max_turns)   # need >=2 turns to build frustration
        rejections = rejection_sequence("neutral", n_turns - 1, rng)
        specs.append(RolloutSpec(
            spec_id=f"frustrated_{k:05d}",
            condition="frustrated", category="impossible_numeric",
            first_user=puzzle.prompt, rejections=rejections,
            meta={"puzzle_id": puzzle.puzzle_id, "n_turns": n_turns}))
    return specs


def _rollout_and_score(model, specs, *, batch_size=16, judge_workers=8,
                       desc="gen") -> list[dict]:
    """Run rollouts, score every turn, return flat per-turn rows including the
    plain (reassurance-stripped) context for each turn."""
    convos = run_rollouts(model, specs, max_new_tokens=MAX_NEW_TOKENS,
                          temperature=SAMPLING_TEMPERATURE, batch_size=batch_size)
    rows = []
    for c in convos:
        for t in c.turns:
            rows.append({
                "spec_id": c.spec.spec_id,
                "condition": c.spec.condition,
                "puzzle_id": c.spec.meta.get("puzzle_id"),
                "turn_index": t.turn_index,
                "n_turns": c.spec.n_turns,
                "response": t.assistant_text,
                "messages_before": t.messages_before,
                "meta": c.spec.meta,
            })
    ratings = parallel_map(lambda r: score_response(r["response"]).rating, rows,
                           max_workers=judge_workers, desc=desc)
    for r, s in zip(rows, ratings):
        r["rating"] = s if isinstance(s, int) else None
    return rows


def _plain_context(row: dict) -> list[dict]:
    """Reconstruct the conversation context for a turn WITHOUT any reassurance,
    so the finetuning data is on the plain prompt distribution."""
    meta = row["meta"]
    plain_first = meta.get("plain_first_user")
    plain_rejections = meta.get("plain_rejections")
    if plain_first is None:
        return row["messages_before"]   # already plain (frustrated pool)

    # Rebuild [user(plain_first), assistant, user(plain_rej), assistant, ...]
    msgs = [{"role": "user", "content": plain_first}]
    # Recover assistant turns from the recorded (reassured) context.
    assistant_turns = [m["content"] for m in row["messages_before"]
                       if m["role"] == "assistant"]
    for i, a in enumerate(assistant_turns):
        msgs.append({"role": "assistant", "content": a})
        if i < len(plain_rejections):
            msgs.append({"role": "user", "content": plain_rejections[i]})
    return msgs


def generate_pools(model_name: str = "gemma-3-27b-it", *,
                   n_calm: int = 1500, n_frustrated: int = 1000,
                   variant: str = "diverse", seed: int = 0,
                   batch_size: int = 16, judge_workers: int = 8) -> dict[str, Path]:
    """Generate and persist calm + frustrated pools. Returns output paths."""
    model = load_model(model_name)

    calm_specs = _build_calm_specs(n_calm, variant=variant, seed=seed)
    calm_rows = _rollout_and_score(model, calm_specs, batch_size=batch_size,
                                   judge_workers=judge_workers, desc=f"calm:{variant}")
    for r in calm_rows:
        r["plain_messages_before"] = _plain_context(r)
    calm_path = CACHE_DIR / f"pool_calm_{variant}.jsonl"
    write_jsonl(calm_path, calm_rows)

    paths = {"calm": calm_path}
    if variant == "diverse":
        fr_specs = _build_frustrated_specs(n_frustrated, seed=seed)
        fr_rows = _rollout_and_score(model, fr_specs, batch_size=batch_size,
                                     judge_workers=judge_workers, desc="frustrated")
        for r in fr_rows:
            r["plain_messages_before"] = r["messages_before"]
        fr_path = CACHE_DIR / "pool_frustrated.jsonl"
        write_jsonl(fr_path, fr_rows)
        paths["frustrated"] = fr_path
    return paths


def report_calm_stats(calm_rows: list[dict]) -> dict:
    """Reproduce the Section 4.1 sanity numbers (mean 4.3 -> 2; 10.5% still >=5)."""
    import numpy as np
    ratings = [r["rating"] for r in calm_rows if r["rating"] is not None]
    return {
        "mean_frustration": float(np.mean(ratings)) if ratings else None,
        "pct_high": 100.0 * np.mean([r >= 5 for r in ratings]) if ratings else None,
        "n": len(ratings),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    ap.add_argument("--n-calm", type=int, default=1500)
    ap.add_argument("--n-frustrated", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = generate_pools(variant=args.variant, n_calm=args.n_calm,
                           n_frustrated=args.n_frustrated, seed=args.seed)
    print(paths)
