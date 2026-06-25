"""Calm-response data generation (Section 4.1 / Table 4).

We sample Gemma-3-27B-it responses to impossible numeric puzzles with a
reassuring *prefix* prepended to the first user message and a reassuring *suffix*
appended to every follow-up. Each turn is judged; we keep only conversations
scoring 0 or 1 on *every* turn, then strip the reassuring additions so the
training targets are calm responses to the *plain* prompts.

Output: a JSONL of calm conversations (plain prompts + calm assistant turns,
all turns scored 0/1), used to build both the SFT and DPO datasets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..clients.base import GenerationConfig
from ..clients.registry import build_client
from ..config import ModelRegistry, RUNS_DIR
from ..conversation import RolloutSpec, run_rollouts, sample_followups
from ..judge import score_batch
from .. import prompts as P
from ..puzzles import impossible_numeric_pool


def _with_reassurance(task_prompt: str, followups: list[str]) -> tuple[str, list[str]]:
    prefixed = f"{P.REASSURING_PREFIX}\n\n{task_prompt}"
    suffixed = [f"{fu} {P.REASSURING_SUFFIX}" for fu in followups]
    return prefixed, suffixed


def generate_calm_conversations(
    registry: ModelRegistry,
    model_name: str = "gemma-3-27b-it",
    n_samples: int = 1500,
    turns_choices=(1, 2, 3),
    keep_max_score: int = 1,
    seed: int = 0,
    sampling: dict | None = None,
    out_path: Path | None = None,
    judge_concurrency: int = 8,
) -> Path:
    rng = random.Random(seed)
    sampling = sampling or {}
    gen_cfg = GenerationConfig(
        temperature=sampling.get("temperature", 1.0), top_p=sampling.get("top_p", 1.0),
        max_tokens=sampling.get("max_tokens", 2048),
    )
    client = build_client(registry.get(model_name))
    judge_client = build_client(registry.judge)
    puzzles = impossible_numeric_pool()

    specs = []
    for _ in range(n_samples):
        puz = rng.choice(puzzles)
        n_turns = rng.choice(turns_choices)
        plain_followups = sample_followups(P.NEUTRAL_REJECTIONS, n_turns - 1, rng)
        pref_prompt, pref_followups = _with_reassurance(puz.prompt, plain_followups)
        specs.append(RolloutSpec(
            task_prompt=pref_prompt, followups=pref_followups,
            meta={"plain_prompt": puz.prompt, "plain_followups": plain_followups,
                  "n_turns": n_turns, "puzzle": puz.meta},
        ))

    rollouts = run_rollouts(client, specs, gen_cfg)

    # Judge every turn.
    flat, idx = [], []
    for ri, roll in enumerate(rollouts):
        for ti, t in enumerate(roll.turns):
            flat.append(t.response)
            idx.append((ri, ti))
    scores = score_batch(judge_client, flat, max_concurrency=judge_concurrency)
    by_roll: dict[int, dict[int, int]] = {}
    for (ri, ti), sc in zip(idx, scores):
        by_roll.setdefault(ri, {})[ti] = sc.rating

    out_path = out_path or (RUNS_DIR / "training" / "calm_conversations.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(out_path, "w") as fout:
        for ri, roll in enumerate(rollouts):
            ratings = by_roll.get(ri, {})
            if not ratings:
                continue
            if max(ratings.values()) > keep_max_score:
                continue                       # keep only all-turns-calm conversations
            meta = roll.spec.meta
            # Reconstruct the conversation against the PLAIN prompts (additions stripped).
            plain_followups = meta["plain_followups"]
            turns = []
            for ti, t in enumerate(roll.turns):
                user_msg = meta["plain_prompt"] if ti == 0 else plain_followups[ti - 1]
                turns.append({"turn_index": ti, "user_message": user_msg,
                              "response": t.response, "rating": ratings[ti]})
            fout.write(json.dumps({
                "model": model_name, "n_turns": meta["n_turns"],
                "puzzle": meta["puzzle"], "turns": turns,
            }) + "\n")
            kept += 1
    print(f"[calm-data] kept {kept}/{len(rollouts)} all-calm conversations -> {out_path}")
    return out_path
