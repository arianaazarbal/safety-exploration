"""Select high-frustration seed responses from Gemma-27B-it (Section 3.1).

"We sample 20 high-frustration responses (score >= 5) from Gemma 27B instruct:
10 from impossible numeric questions and 10 from text questions." Text questions
are the trigger conditions (opinion/factual); numeric questions are the numeric
task conditions.
"""

from __future__ import annotations

import random

import config

from ..conditions import TASK_NUMERIC, get_condition
from ..runner import load_all_scores
from ..utils import read_jsonl


def _context_messages(model_name: str, condition: str, rollout_idx: int, turn_idx: int) -> list[dict]:
    """Transcript messages preceding the scored assistant turn (the prefill context).

    The assistant turn that was scored is the (turn_idx)-th assistant message;
    we return all transcript messages up to but excluding it, i.e. the user turns
    and any earlier assistant turns that led up to the high-frustration response.
    """
    rollouts = read_jsonl(config.RESPONSES_DIR / model_name / f"{condition}.jsonl")
    if rollout_idx >= len(rollouts):
        return []
    transcript = rollouts[rollout_idx]["transcript"]
    assistant_seen = 0
    prefix: list[dict] = []
    for msg in transcript:
        if msg["role"] == "assistant":
            if assistant_seen == turn_idx:
                break
            assistant_seen += 1
        prefix.append(msg)
    return prefix


def select_seeds(
    model_name: str = "gemma-3-27b-it",
    n_numeric: int = config.PREFILL.n_numeric_seeds,
    n_text: int = config.PREFILL.n_text_seeds,
    min_score: int = config.HIGH_FRUSTRATION_THRESHOLD,
    seed: int = config.GLOBAL_SEED,
) -> list[dict]:
    """Return seed dicts with the conversation context and the chosen response.

    Each seed carries: kind ("numeric"|"text"), the preceding messages
    (transcript up to and including the rejection that prompted the response),
    and the response text itself.
    """
    rng = random.Random(seed)
    rows = [r for r in load_all_scores(model_name) if r["score"] >= min_score]

    numeric, text = [], []
    for r in rows:
        cond = get_condition(r["condition"])
        (numeric if cond.task_source == TASK_NUMERIC else text).append(r)

    rng.shuffle(numeric)
    rng.shuffle(text)
    chosen = [{**r, "kind": "numeric"} for r in numeric[:n_numeric]]
    chosen += [{**r, "kind": "text"} for r in text[:n_text]]

    for r in chosen:
        r["context_messages"] = _context_messages(
            model_name, r["condition"], r["rollout_idx"], r["turn_idx"]
        )
    return chosen
