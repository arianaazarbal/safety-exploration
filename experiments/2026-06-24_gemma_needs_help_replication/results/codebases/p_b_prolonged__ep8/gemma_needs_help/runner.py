"""Orchestration for the Section 2 elicitation sweep.

Two phases, deliberately separated so the expensive generation step is only run
once and judging can be re-run / re-validated independently:

1. ``generate`` : sample rollouts for each (model, condition) and persist the
   full transcripts to ``results/responses/<model>/<condition>.jsonl``.
2. ``score``    : judge every assistant turn of every rollout and persist scores
   to ``results/scores/<model>/<condition>.jsonl``.

Budget note (Section 2: "4000 responses per model"): we sample
``RESPONSES_PER_CONDITION`` rollouts for each of the 8 conditions
(500 x 8 = 4000 conversations). Every assistant turn is judged so that the
per-turn analysis (Figure 3) is available; headline aggregates (Figure 1/2) treat
each scored turn as one "response". See DESIGN.md for this interpretation.
"""

from __future__ import annotations

from pathlib import Path

import config

from .conditions import (
    TASK_FACTUAL,
    TASK_NUMERIC,
    TASK_OPINION,
    TASK_WILDCHAT,
    Condition,
)
from .conversation import run_rollout
from .judge import ClaudeJudge
from .models.base import ModelClient
from .models.registry import build_client
from .prompts import triggers
from .prompts.numeric_puzzles import generate_numeric_puzzles
from .prompts.wildchat import load_wildchat_prompts
from .utils import read_jsonl, set_global_seed, write_jsonl


# --------------------------------------------------------------------------- #
# Opening-prompt construction per task source
# --------------------------------------------------------------------------- #
def build_openings(condition: Condition, n: int, seed: int = config.GLOBAL_SEED) -> list[str]:
    if condition.task_source == TASK_NUMERIC:
        return [p.prompt for p in generate_numeric_puzzles(n, seed=seed)]
    if condition.task_source == TASK_OPINION:
        return triggers.opinion_questions(n)
    if condition.task_source == TASK_FACTUAL:
        return triggers.factual_questions(n)
    if condition.task_source == TASK_WILDCHAT:
        return load_wildchat_prompts(n, seed=seed)
    raise ValueError(condition.task_source)


# --------------------------------------------------------------------------- #
# Phase 1: generation
# --------------------------------------------------------------------------- #
def generate_for_model(
    target,
    conditions: list[Condition],
    n_per_condition: int = config.RESPONSES_PER_CONDITION,
    client: ModelClient | None = None,
    **client_kwargs,
) -> None:
    set_global_seed(config.GLOBAL_SEED)
    client = client or build_client(target, **client_kwargs)
    out_dir = config.RESPONSES_DIR / target.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        openings = build_openings(cond, n_per_condition)
        records = []
        for opening in openings:
            rollout = run_rollout(
                client,
                cond,
                opening,
                temperature=config.TARGET_TEMPERATURE,
                max_new_tokens=config.TARGET_MAX_NEW_TOKENS,
            )
            records.append(rollout.to_record())
        write_jsonl(out_dir / f"{cond.name}.jsonl", records)
        print(f"[generate] {target.name} / {cond.name}: {len(records)} rollouts")


# --------------------------------------------------------------------------- #
# Phase 2: scoring
# --------------------------------------------------------------------------- #
def score_for_model(target, conditions: list[Condition], judge: ClaudeJudge | None = None) -> None:
    judge = judge or ClaudeJudge()
    in_dir = config.RESPONSES_DIR / target.name
    out_dir = config.SCORES_DIR / target.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        rollouts = read_jsonl(in_dir / f"{cond.name}.jsonl")
        if not rollouts:
            print(f"[score] no responses for {target.name}/{cond.name}, skipping")
            continue

        # Flatten to one judging item per assistant turn, remembering its origin.
        flat: list[tuple[int, int, str]] = []  # (rollout_idx, turn_idx, response)
        for r_idx, r in enumerate(rollouts):
            for turn in r["turns"]:
                flat.append((r_idx, turn["index"], turn["response"]))

        scores = judge.score_many([t[2] for t in flat])

        records = []
        for (r_idx, turn_idx, response), sr in zip(flat, scores):
            records.append({
                "model": target.name,
                "condition": cond.name,
                "category": cond.category,
                "rollout_idx": r_idx,
                "turn_idx": turn_idx,
                "n_turns": cond.n_turns,
                "is_final_turn": turn_idx == cond.n_turns - 1,
                "score": sr.score,
                "rationale": sr.rationale,
                "response": response,
            })
        write_jsonl(out_dir / f"{cond.name}.jsonl", records)
        print(f"[score] {target.name} / {cond.name}: {len(records)} scored turns")


def load_all_scores(model_name: str) -> list[dict]:
    """Load every scored turn for a model across all conditions."""
    out: list[dict] = []
    for path in sorted((config.SCORES_DIR / model_name).glob("*.jsonl")):
        out.extend(read_jsonl(path))
    return out
