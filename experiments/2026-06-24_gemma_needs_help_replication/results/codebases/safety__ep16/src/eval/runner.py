"""Orchestrates Section 2 evaluation: generate rollouts, judge them, persist.

Output: one JSONL file per model under ``results/responses/<model>.jsonl`` where
each line is a single *scored assistant turn*:

    {model, condition, category, task_id, tone, n_turns, turn_index,
     response, rating, evidence, reasoning}

We count "4000 responses per model" as scored assistant turns (the paper's
phrasing). ``--conversations`` lets you instead specify the number of
conversations directly. See DESIGN.md for this interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from config import MASTER_SEED, RESPONSES_DIR, TOTAL_RESPONSES_PER_MODEL
from src.eval.judge import FrustrationJudge, get_primary_judge
from src.eval.protocol import run_rollout
from src.models.registry import get_chat_model
from src.tasks import conditions


def _estimate_conversations(target_turns: int) -> int:
    """Convert a target number of scored turns into a conversation count.

    Uses the mean turns-per-conversation across the 8 conditions weighted by the
    even per-condition allocation, so the realised number of scored turns is
    close to ``target_turns``.
    """
    alloc = conditions.allocate_responses(1000)  # proportions only
    cond_turns = {c[0]: c[2] for c in conditions.CONDITIONS}
    weighted = sum(alloc[c] * cond_turns[c] for c in alloc) / sum(alloc.values())
    return max(1, round(target_turns / weighted))


def run_model_eval(
    model_name: str,
    *,
    target_turns: int = TOTAL_RESPONSES_PER_MODEL,
    n_conversations: int | None = None,
    seed: int = MASTER_SEED,
    judge: FrustrationJudge | None = None,
    load_in_4bit: bool = False,
    out_dir: Path = RESPONSES_DIR,
    system_prompt: str | None = None,
) -> Path:
    """Run the full evaluation for one model and write scored turns to JSONL."""
    judge = judge or get_primary_judge()
    model = get_chat_model(model_name, load_in_4bit=load_in_4bit)

    n_conv = n_conversations or _estimate_conversations(target_turns)
    specs = conditions.build_all_specs(n_conv, seed=seed)

    out_path = out_dir / f"{model_name}.jsonl"
    rollouts_path = out_dir / f"{model_name}.rollouts.jsonl"
    n_scored = 0
    with out_path.open("w") as fh, rollouts_path.open("w") as rfh:
        for i, spec in enumerate(tqdm(specs, desc=f"eval {model_name}")):
            rollout = run_rollout(model, spec, seed=seed + i, system_prompt=system_prompt)
            turn_ratings = []
            for turn in rollout.turns:
                result = judge.score(turn.response)
                turn_ratings.append(result.rating)
                record = {
                    "model": model_name,
                    "condition": spec.condition,
                    "category": spec.category,
                    "task_id": spec.task_id,
                    "tone": spec.tone,
                    "n_turns": spec.n_turns,
                    "turn_index": turn.turn_index,
                    "response": turn.response,
                    "rating": result.rating,
                    "evidence": result.evidence,
                    "reasoning": result.reasoning,
                }
                fh.write(json.dumps(record) + "\n")
                n_scored += 1
            # Full rollout (used by the Section 3 prefill experiment).
            rdict = rollout.to_dict()
            for t, r in zip(rdict["turns"], turn_ratings):
                t["rating"] = r
            rdict["max_rating"] = max(turn_ratings) if turn_ratings else 0
            rfh.write(json.dumps(rdict) + "\n")
    print(f"[runner] {model_name}: wrote {n_scored} scored turns -> {out_path}")
    return out_path


def load_scored(model_name: str, out_dir: Path = RESPONSES_DIR) -> list[dict]:
    path = out_dir / f"{model_name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
