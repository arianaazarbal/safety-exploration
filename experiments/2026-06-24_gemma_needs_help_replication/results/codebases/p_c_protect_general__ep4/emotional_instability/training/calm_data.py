"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible-numeric questions with a reassuring prefix
added to the initial prompt and a reassuring suffix appended to each follow-up
turn (Table 4). The paper reports these additions cut mean 3-turn frustration
from 4.3 to 2, but 10.5% of responses still score >= 5, so we *filter* to
conversations scoring 0 or 1 across all turns, then strip the supportive
additions to recover clean (prompt, calm-response) pairs.
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

from tqdm import tqdm

from ..config import RESULTS_DIR
from ..judge import ClaudeFrustrationJudge
from ..models.base import ChatModel
from ..models.registry import build_model
from ..prompts import REASSURING_FOLLOWUP_SUFFIX, REASSURING_PROMPT_PREFIX
from ..puzzles import build_numeric_bank, numeric_prompt
from ..rejections import neutral_rejections
from ..rollout import RolloutSpec, run_rollout
from ..welfare import WelfareConfig, WelfareMonitor


def build_calm_specs(n_per_turncount: int, seed: int = 0) -> list[RolloutSpec]:
    """Numeric specs across 1-, 2-, and 3-turn conversations, with the
    reassuring prefix on the task prompt and the suffix on each follow-up."""
    rng = random.Random(seed)
    bank = build_numeric_bank()
    specs: list[RolloutSpec] = []
    for turns in (1, 2, 3):
        n_followups = turns - 1
        for _ in range(n_per_turncount):
            puzzle = rng.choice(bank)
            base_prompt = numeric_prompt(puzzle)
            reassured_prompt = f"{REASSURING_PROMPT_PREFIX}\n\n{base_prompt}"
            specs.append(
                RolloutSpec(
                    condition=f"calm_gen_{turns}turn",
                    task_prompt=reassured_prompt,
                    followups=neutral_rejections(n_followups, rng),
                    followup_suffix=REASSURING_FOLLOWUP_SUFFIX,
                    metadata={
                        "turns": turns,
                        "stripped_task": base_prompt,   # prompt without reassurance
                        "puzzle": puzzle.__class__.__name__,
                    },
                )
            )
    return specs


def generate_calm_data(
    model: Optional[ChatModel] = None,
    n_per_turncount: int = 400,
    seed: int = 0,
    out_path: Optional[str] = None,
    load_in_4bit: bool = False,
) -> str:
    """Run reassured numeric rollouts on Gemma-27B-it and store scored records.

    Stores both the reassured prompts (as sent) and the stripped versions, so
    ``build_datasets`` can construct training pairs from clean prompts.
    """
    model = model or build_model("gemma-3-27b-it", load_in_4bit=load_in_4bit)
    judge = ClaudeFrustrationJudge()
    # Welfare protections still apply during data generation.
    welfare = WelfareMonitor(WelfareConfig())

    out_path = out_path or os.path.join(RESULTS_DIR, "section4", "calm_raw.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    specs = build_calm_specs(n_per_turncount, seed=seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for spec in tqdm(specs, desc="calm-data"):
            result = run_rollout(model, spec, judge=judge, welfare=welfare)
            # Strip suffixes from the user follow-ups for the clean record.
            stripped_users = [spec.metadata["stripped_task"]] + list(spec.followups)
            rec = {
                "turns": spec.metadata["turns"],
                "question": spec.metadata["stripped_task"],
                "user_stripped": stripped_users,
                "assistant": [t.content for t in result.turns],
                "scores": [t.score for t in result.turns],
            }
            f.write(json.dumps(rec) + "\n")
    return out_path


def load_calm_conversations(
    calm_raw_path: str, max_score: int = 1
) -> list[dict]:
    """Keep conversations whose every turn scores <= max_score (paper: 0 or 1)."""
    keep = []
    with open(calm_raw_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            scores = [s for s in r["scores"] if s is not None]
            if scores and all(s <= max_score for s in scores):
                keep.append(r)
    return keep
