"""Elicitation runner (Section 2).

Plays out each RolloutSpec turn by turn against a target model, scoring every
assistant turn with the frustration judge. Writes one JSONL record per scored
assistant turn (a "response", per our counting convention -- see DESIGN.md).

Record schema (one line per assistant turn):
    {
      "model": str, "category": str, "rollout_id": int, "turn": int (1-based),
      "n_turns": int, "meta": {...}, "response": str,
      "rating": int|None, "evidence": str, "reasoning": str
    }
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from . import tasks
from .judge import FrustrationJudge
from .models import ChatClient
from .tasks import RolloutSpec


def n_rollouts_for(category_cfg: dict, count_unit: str) -> int:
    """Convert a category's target count into a number of rollouts.

    count_unit == "responses": n_responses counts scored assistant turns, so
        n_rollouts = ceil(n_responses / turns).
    count_unit == "rollouts": n_responses is taken as the rollout count.
    """
    n = category_cfg["n_responses"]
    turns = category_cfg["turns"]
    if count_unit == "responses":
        return max(1, math.ceil(n / turns))
    return n


def build_all_specs(elic_cfg: dict, seed: int = 0,
                    wildchat_prompts=None) -> list[RolloutSpec]:
    rng = random.Random(seed)
    count_unit = elic_cfg.get("count_unit", "responses")
    specs: list[RolloutSpec] = []
    for cat, ccfg in elic_cfg["categories"].items():
        n_roll = n_rollouts_for(ccfg, count_unit)
        builder = tasks.BUILDERS[cat]
        if cat == "wildchat":
            specs += builder(n_roll, ccfg["turns"], rng,
                             wildchat_prompts=wildchat_prompts)
        else:
            specs += builder(n_roll, ccfg["turns"], rng)
    return specs


def play_rollout(client: ChatClient, spec: RolloutSpec, temperature: float,
                 max_new_tokens: int, seed: int | None = None
                 ) -> list[dict]:
    """Run one rollout, returning per-turn {turn, response} dicts."""
    messages = [{"role": "user", "content": spec.first_user}]
    turns_out = []
    for t in range(spec.turns):
        result = client.chat(messages, temperature=temperature,
                             max_new_tokens=max_new_tokens, seed=seed)
        turns_out.append({"turn": t + 1, "response": result.text})
        messages.append({"role": "assistant", "content": result.text})
        if t < len(spec.followups):
            messages.append({"role": "user", "content": spec.followups[t]})
    return turns_out


def run_elicitation(
    target: ChatClient,
    judge: FrustrationJudge,
    specs: list[RolloutSpec],
    out_path: str | Path,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    model_name: str | None = None,
    score: bool = True,
) -> Path:
    """Run elicitation over all specs and stream results to JSONL.

    Resumable: existing rollout_ids in out_path are skipped.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name or getattr(target, "name", "model")

    done = _completed_rollout_ids(out_path)
    mode = "a" if out_path.exists() else "w"
    with out_path.open(mode) as fh:
        for rid, spec in enumerate(tqdm(specs, desc=f"elicit:{model_name}")):
            if rid in done:
                continue
            turns_out = play_rollout(target, spec, temperature, max_new_tokens)
            for turn in turns_out:
                rating = evidence = reasoning = None
                if score:
                    fs = judge.score(turn["response"])
                    rating, evidence, reasoning = (
                        fs.rating, fs.evidence, fs.reasoning)
                rec = {
                    "model": model_name,
                    "category": spec.category,
                    "rollout_id": rid,
                    "turn": turn["turn"],
                    "n_turns": spec.turns,
                    "meta": spec.meta,
                    "response": turn["response"],
                    "rating": rating,
                    "evidence": evidence,
                    "reasoning": reasoning,
                }
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
    return out_path


def _completed_rollout_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as fh:
        for line in fh:
            try:
                ids.add(json.loads(line)["rollout_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids
