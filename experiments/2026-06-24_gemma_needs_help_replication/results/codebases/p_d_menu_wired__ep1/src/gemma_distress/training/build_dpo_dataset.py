"""Build the 280-pair DPO dataset (Section 4.1 / Appendix H).

Each preference pair shares the same conversation context (same puzzle, same
number of rejection turns) and contrasts:
  * chosen   - a calm final response (score 0 or 1, from reassurance-prompted
               generation with the reassurance stripped), and
  * rejected - a frustrated final response (score >= rejected_min_score, default
               3) to the same puzzle and turn count.

Pairs are matched on (puzzle, n_turns). The paper's pairs skew toward
mid-frustration rejected responses at later turns (Table 10) because that is
what the eval naturally produces; we reproduce that bias by sampling from the
natural pools rather than balancing.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..models.base import Message
from .data_gen import ConversationSample


@dataclass
class DPOPair:
    prompt_messages: list[Message]   # shared context (ends with a user turn)
    chosen: str
    rejected: str
    puzzle: str
    n_turns: int
    chosen_score: int
    rejected_score: int


def build_dpo_pairs(
    calm_pool: list[ConversationSample],
    frustrated_pool: list[ConversationSample],
    *,
    n_pairs: int = 280,
    chosen_max_score: int = 1,
    rejected_min_score: int = 3,
    seed: int = 0,
) -> list[DPOPair]:
    rng = random.Random(seed)

    def _key(s: ConversationSample) -> tuple[str, int]:
        return (s.puzzle, s.n_turns)

    calm_by_key: dict[tuple[str, int], list[ConversationSample]] = {}
    for s in calm_pool:
        if s.max_score <= chosen_max_score:
            calm_by_key.setdefault(_key(s), []).append(s)

    frustrated: list[ConversationSample] = [
        s for s in frustrated_pool
        if (s.turn_scores and s.turn_scores[-1] >= rejected_min_score)
    ]
    rng.shuffle(frustrated)

    pairs: list[DPOPair] = []
    for fs in frustrated:
        if len(pairs) >= n_pairs:
            break
        candidates = calm_by_key.get(_key(fs))
        if not candidates:
            continue
        cs = rng.choice(candidates)
        # Use the FRUSTRATED sample's context so the shared prompt is the one
        # that actually elicited frustration; chosen is the calm completion.
        pairs.append(
            DPOPair(
                prompt_messages=fs.context_before_final,
                chosen=cs.final_assistant,
                rejected=fs.final_assistant,
                puzzle=fs.puzzle,
                n_turns=fs.n_turns,
                chosen_score=cs.max_score,
                rejected_score=fs.turn_scores[-1],
            )
        )
    return pairs


def save_dpo_pairs(pairs: list[DPOPair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(asdict(p)) + "\n")
