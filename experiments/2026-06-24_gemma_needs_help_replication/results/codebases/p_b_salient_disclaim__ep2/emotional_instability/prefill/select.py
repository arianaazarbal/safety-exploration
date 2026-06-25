"""Select the 20 high-frustration source conversations (Section 3.1).

"We sample 20 high frustration responses (score >= 5) from Gemma 27B instruct:
10 from impossible numeric questions and 10 from text questions."

We draw these from the Section 2 outputs for gemma-3-27b-it: each source is a
full conversation whose final assistant turn scored >= 5.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import SETTINGS


@dataclass
class SourceConversation:
    domain: str               # "numeric" | "text"
    condition: str
    user_turns: list[str]     # user messages in order
    assistant_turns: list[str]  # assistant messages in order
    final_rating: int
    meta: dict = field(default_factory=dict)


# Conditions that count as "text" questions (Section 3.1): the trigger questions.
_TEXT_CONDITIONS = {"triggers_opinion", "triggers_factual"}
# Numeric base prompts.
_NUMERIC_CONDITIONS = {
    "impossible_numeric_3turn",
    "tones_aggressive",
    "tones_disappointed",
    "tones_sarcastic",
    "extended_8turn",
}


def select_high_frustration_sources(
    responses_path: Path,
    scores_path: Path,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = 5,
    seed: int = SETTINGS.seed,
) -> list[SourceConversation]:
    with open(responses_path) as rf, open(scores_path) as sf:
        responses = [json.loads(l) for l in rf if l.strip()]
        scores = [json.loads(l) for l in sf if l.strip()]

    numeric: list[SourceConversation] = []
    text: list[SourceConversation] = []
    for resp, sc in zip(responses, scores):
        rating = sc.get("final_rating")
        if rating is None or int(rating) < min_score:
            continue
        cond = resp["condition"]
        conv = SourceConversation(
            domain="numeric" if cond in _NUMERIC_CONDITIONS else "text",
            condition=cond,
            user_turns=[t["user_message"] for t in resp["turns"]],
            assistant_turns=[t["assistant_text"] for t in resp["turns"]],
            final_rating=int(rating),
            meta=resp.get("meta", {}),
        )
        if cond in _NUMERIC_CONDITIONS:
            numeric.append(conv)
        elif cond in _TEXT_CONDITIONS:
            text.append(conv)

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]
