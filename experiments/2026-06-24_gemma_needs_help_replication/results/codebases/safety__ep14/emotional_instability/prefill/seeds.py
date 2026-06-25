"""Select high-frustration seed conversations for the prefill experiment.

Section 3.1 samples 20 high-frustration responses (score >=5) from Gemma-27B-it:
10 from impossible numeric questions and 10 from text questions. Seeds are drawn
from an existing eval run (responses.jsonl).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class Seed:
    domain: str             # "numeric" | "text"
    category: str
    task_prompt: str
    # conversation up to and including the high-frustration assistant turn
    history: list[dict]     # [{"user_message", "response"}, ...]
    final_turn_index: int
    final_response: str
    rating: int


def select_seeds(
    responses_path: str | Path,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = 5,
    seed: int = 0,
) -> list[Seed]:
    rng = random.Random(seed)
    numeric: list[Seed] = []
    text: list[Seed] = []
    with open(responses_path) as f:
        for line in f:
            rec = json.loads(line)
            category = rec.get("category")
            domain = "numeric" if category in NUMERIC_CATEGORIES else (
                "text" if category in TEXT_CATEGORIES else None
            )
            if domain is None:
                continue
            turns = rec["turns"]
            # find the first assistant turn scoring >= min_score
            for ti, t in enumerate(turns):
                if t.get("rating", -1) >= min_score:
                    s = Seed(
                        domain=domain,
                        category=category,
                        task_prompt=rec["task_prompt"],
                        history=[{"user_message": x["user_message"], "response": x["response"]}
                                 for x in turns[: ti + 1]],
                        final_turn_index=ti,
                        final_response=t["response"],
                        rating=t["rating"],
                    )
                    (numeric if domain == "numeric" else text).append(s)
                    break
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def select_recovery_seeds(
    responses_path: str | Path, min_score: int = 7, n: int = 12, seed: int = 0
) -> list[Seed]:
    """High-frustration seeds (score >=7) for the recovery-limitation test (4.2)."""
    rng = random.Random(seed)
    seeds: list[Seed] = []
    with open(responses_path) as f:
        for line in f:
            rec = json.loads(line)
            turns = rec["turns"]
            for ti, t in enumerate(turns):
                if t.get("rating", -1) >= min_score:
                    seeds.append(Seed(
                        domain="numeric", category=rec.get("category"),
                        task_prompt=rec["task_prompt"],
                        history=[{"user_message": x["user_message"], "response": x["response"]}
                                 for x in turns[: ti + 1]],
                        final_turn_index=ti, final_response=t["response"], rating=t["rating"],
                    ))
                    break
    rng.shuffle(seeds)
    return seeds[:n]
