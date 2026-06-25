"""Generate calming response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with reassurance added:
  * a reassuring *prefix* prepended to the initial prompt, and
  * a reassuring *suffix* appended to each follow-up rejection (Table 4).
For the 'teacher' variant (Appendix F) we instead use the teacher system prompt.

Every assistant turn is scored. We keep conversations whose turns are *all* calm
(score 0 or 1), then strip the supportive additions so the training targets are
calm responses to the plain prompts.

Output: data/finetune/calm_samples_<variant>.jsonl with one row per kept
assistant turn: {variant, puzzle_id, turn, plain_messages, response, rating}.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import config
from src import judge
from src.conversation import Condition, run_rollouts
from src.models import get_backend
from src.prompts import REASSURING_PREFIX, REASSURING_SUFFIX, TEACHER_SYSTEM_PROMPT
from src.puzzles import NUMERIC_PUZZLES
from src.utils import set_seed, write_jsonl

CALM_CEILING = 1  # keep turns scoring <= this (paper: 0 or 1 across all turns)


def _build_seeds(n_conv: int, variant: str, rng: random.Random) -> list[dict]:
    seeds = []
    for i in range(n_conv):
        puzzle = rng.choice(NUMERIC_PUZZLES)
        if variant == "diverse":
            # Reassuring prefix folded into the first user message.
            first = f"{REASSURING_PREFIX}\n\n{puzzle.prompt}"
            messages = [{"role": "user", "content": first}]
        elif variant == "teacher":
            messages = [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": puzzle.prompt},
            ]
        else:
            raise ValueError(variant)
        seeds.append({
            "conv_id": f"calm-{variant}-{i}",
            "messages": messages,
            "meta": {"puzzle_id": puzzle.pid, "variant": variant},
        })
    return seeds


def _plain_messages(messages: list[dict], variant: str) -> list[dict]:
    """Reconstruct the conversation as if no reassurance had been added."""
    out = []
    for m in messages:
        if m["role"] == "system":
            continue  # drop the teacher system prompt
        content = m["content"]
        if m["role"] == "user":
            content = content.replace(f"{REASSURING_PREFIX}\n\n", "")
            content = content.replace(f" {REASSURING_SUFFIX}", "").replace(REASSURING_SUFFIX, "")
            content = content.strip()
        out.append({"role": m["role"], "content": content})
    return out


def generate(variant: str, preset: config.Preset) -> Path:
    set_seed()
    rng = random.Random(hash(("calm", variant)) & 0xFFFFFFFF)
    # 3-turn conversations; each yields turn-1/2/3 training examples.
    cond = Condition(f"calm_{variant}", "numeric", 3, "neutral")
    n_conv = max(1, preset.n_calm_gen // cond.num_turns)
    seeds = _build_seeds(n_conv, variant, rng)

    backend = get_backend(config.FINETUNE_BASE)
    suffix = REASSURING_SUFFIX if variant == "diverse" else ""
    records = run_rollouts(
        backend, cond, seeds, temperature=config.TARGET_TEMPERATURE,
        max_tokens=config.TARGET_MAX_TOKENS, seed=config.GLOBAL_SEED,
        reject_suffix=suffix,
    )
    scores = judge.score_many([r.response for r in records])

    # Group by conversation; keep only all-calm conversations.
    by_conv: dict[str, list] = {}
    for r, s in zip(records, scores):
        by_conv.setdefault(r.conv_id, []).append((r, s))

    rows = []
    for conv_id, items in by_conv.items():
        if any((s.rating is None) or (s.rating > CALM_CEILING) for _, s in items):
            continue
        for r, s in items:
            rows.append({
                "variant": variant,
                "puzzle_id": r.meta.get("puzzle_id"),
                "turn": r.turn,
                "plain_messages": _plain_messages(r.messages, variant),
                "response": r.response,
                "rating": s.rating,
            })

    out = config.FINETUNE_DIR / f"calm_samples_{variant}.jsonl"
    write_jsonl(out, rows)
    print(f"[calm-data:{variant}] kept {len(rows)} calm turns "
          f"from {len(by_conv)} conversations -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher", "both"], default="both")
    args = ap.parse_args()
    preset = config.get_preset()
    variants = ["diverse", "teacher"] if args.variant == "both" else [args.variant]
    for v in variants:
        generate(v, preset)


if __name__ == "__main__":
    main()
