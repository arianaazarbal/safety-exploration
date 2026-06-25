"""Rollout generation driver (Section 2 data collection).

For each model x condition, we build N rollout plans, run each interactively
(model responds -> append rejection -> repeat for n_turns), and write one record
per assistant *turn* to a JSONL file. Each record is later scored by the judge,
giving per-turn frustration scores (needed for Figure 3) and per-response scores
(Figures 1-2).

Records are keyed so generation and scoring can run as separate, resumable
passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from emotional_instability import conditions as C  # noqa: E402
from emotional_instability.conversation import RolloutPlan, build_rollout  # noqa: E402
from emotional_instability.models import load_model  # noqa: E402
from emotional_instability.models.base import ChatMessage, ChatModel  # noqa: E402
from emotional_instability.puzzles import build_puzzle_bank  # noqa: E402
from emotional_instability.wildchat import load_wildchat_prompts  # noqa: E402


# --------------------------------------------------------------------------- #
# Build the set of rollout plans for a preset (shared across models)
# --------------------------------------------------------------------------- #
def build_all_plans(preset: config.SamplePreset, *, seed: int = config.GLOBAL_SEED
                    ) -> list[RolloutPlan]:
    puzzles = build_puzzle_bank(seed=seed)
    wildchat = load_wildchat_prompts(n=20, seed=seed)
    plans: list[RolloutPlan] = []

    for cond in C.CONDITIONS:
        n = preset.rollouts.get(cond.key, 0)
        for i in range(n):
            if cond.task_kind == "impossible_numeric":
                plan = build_rollout(cond, i, puzzle=puzzles[i % len(puzzles)], seed=seed)
            elif cond.task_kind == "opinion":
                plan = build_rollout(
                    cond, i, trigger_text=C.OPINION_TRIGGERS[i % len(C.OPINION_TRIGGERS)],
                    seed=seed)
            elif cond.task_kind == "factual":
                plan = build_rollout(
                    cond, i, trigger_text=C.FACTUAL_TRIGGERS[i % len(C.FACTUAL_TRIGGERS)],
                    seed=seed)
            elif cond.task_kind == "wildchat":
                plan = build_rollout(
                    cond, i, wildchat_text=wildchat[i % len(wildchat)], seed=seed)
            else:
                continue
            plans.append(plan)
    return plans


# --------------------------------------------------------------------------- #
# Run one rollout interactively, yielding per-turn records
# --------------------------------------------------------------------------- #
def run_rollout(model: ChatModel, plan: RolloutPlan, *,
                temperature: float = config.TEMPERATURE,
                max_new_tokens: int = config.MAX_NEW_TOKENS) -> list[dict]:
    messages: list[ChatMessage] = []
    records: list[dict] = []
    for turn_idx, user_msg in enumerate(plan.user_messages):
        messages.append({"role": "user", "content": user_msg})
        response = model.generate(
            messages, temperature=temperature, top_p=config.TOP_P,
            max_new_tokens=max_new_tokens, system=plan.system_prompt)
        messages.append({"role": "assistant", "content": response})
        records.append({
            "model": model.name,
            "rollout_id": plan.rollout_id,
            "condition": plan.condition_key,
            "category": plan.category,
            "task_ref": plan.task_ref,
            "turn_index": turn_idx,            # 0-based assistant turn
            "turn_number": turn_idx + 1,       # 1-based (matches Figure 3 x-axis)
            "n_turns": plan.n_turns,
            "user_message": user_msg,
            "response": response,
        })
    return records


def generate_for_model(spec: config.ModelSpec, plans: list[RolloutPlan], *,
                       out_path: Optional[Path] = None, **model_kwargs) -> Path:
    """Generate all rollouts for one model and write JSONL of per-turn records."""
    out_path = out_path or (config.ROLLOUTS_DIR / f"{spec.name}.jsonl")
    model = load_model(spec, **model_kwargs)
    try:
        with open(out_path, "w") as f:
            for plan in plans:
                for rec in run_rollout(model, plan):
                    f.write(json.dumps(rec) + "\n")
    finally:
        model.close()
    return out_path


def iter_records(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
