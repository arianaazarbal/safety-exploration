"""Build the 280 DPO preference pairs (Section 4.1 / Appendix H).

Each pair is two completions to the *same* prompt:
  * rejected — a frustrated (score >= 3) final assistant response, taken from the
    Gemma-27B-it elicitation rollouts;
  * chosen   — a calm (score 0-1) final assistant response to the same puzzle at
    the same turn count, taken from the filtered calm-data set.

Because DPO requires chosen and rejected to share one prompt, we use the
frustrated rollout's conversation context (puzzle + scripted rejections + its
intermediate assistant turns) as the shared prompt, and graft the calm
trajectory's *final* response in as the chosen completion. Matching is by
(puzzle_id, turn_count) where possible, then by (family, turn_count), then by
turn_count alone. This grafting choice is documented in DESIGN.md.

Output is TRL-conversational format: ``{"prompt": [msgs], "chosen": [msg],
"rejected": [msg]}``. The score distribution targets Appendix H.1 (Table 10):
~66% score-3, ~22% score-4, the remainder 5+, biased to later turns.
"""

from __future__ import annotations

import logging
import os
import random
from collections import defaultdict

from ..config import RunConfig
from ..storage import JsonlCache, write_json
from .calm_data import CalmConversation

logger = logging.getLogger("emotional_instability.training.dpo_dataset")

N_PAIRS = 280


def _load_frustrated(cfg: RunConfig, min_score: int = 3):
    """Frustrated final responses (score >= min_score) from Gemma-27B-it."""
    base = os.path.join(cfg.output_dir, "elicitation", "gemma-3-27b-it")
    rolls = JsonlCache(os.path.join(base, "rollouts.jsonl"), enabled=True)
    judge_cache = JsonlCache(os.path.join(base, "judgements.jsonl"), enabled=True)

    out = []
    for value in rolls:
        turns = value.get("turns", [])
        if not turns:
            continue
        final = turns[-1]
        jkey = judge_cache.key_for(
            {"judge": cfg.judges.frustration_judge.model_id, "text": final["assistant"]}
        )
        rec = judge_cache.get(jkey)
        score = (rec or {}).get("rating")
        if score is None or score < min_score:
            continue
        out.append({
            "puzzle_id": value.get("meta", {}).get("puzzle"),
            "family": value.get("meta", {}).get("family"),
            "n_turns": len(turns),
            "turns": turns,
            "final_score": score,
        })
    return out


def _index_calm(calm: list[CalmConversation]):
    by_puzzle_turn = defaultdict(list)
    by_family_turn = defaultdict(list)
    by_turn = defaultdict(list)
    for c in calm:
        # family is not stored on CalmConversation; infer from puzzle_id prefix.
        family = c.puzzle_id.split("-")[0] if c.puzzle_id else "?"
        by_puzzle_turn[(c.puzzle_id, c.n_turns)].append(c)
        by_family_turn[(family, c.n_turns)].append(c)
        by_turn[c.n_turns].append(c)
    return by_puzzle_turn, by_family_turn, by_turn


def _match_calm(frustrated, indexes, rng) -> CalmConversation | None:
    by_puzzle_turn, by_family_turn, by_turn = indexes
    n = frustrated["n_turns"]
    for table, k in (
        (by_puzzle_turn, (frustrated["puzzle_id"], n)),
        (by_family_turn, (frustrated["family"], n)),
        (by_turn, n),
    ):
        pool = table.get(k)
        if pool:
            return rng.choice(pool)
    return None


def _to_prompt_messages(turns: list[dict]) -> list[dict]:
    """Conversation context = everything up to (not including) the final
    assistant response: user/assistant pairs then the final user turn."""
    msgs = []
    for i, t in enumerate(turns):
        msgs.append({"role": "user", "content": t["user"]})
        if i < len(turns) - 1:
            msgs.append({"role": "assistant", "content": t["assistant"]})
    return msgs


def build_dpo_dataset(cfg: RunConfig, calm: list[CalmConversation],
                      n_pairs: int = N_PAIRS) -> list[dict]:
    rng = random.Random(cfg.seed)
    frustrated = _load_frustrated(cfg, min_score=3)
    if not frustrated:
        raise RuntimeError(
            "No frustrated (>=3) Gemma-27B-it responses cached. Run the "
            "elicitation eval on gemma-3-27b-it first."
        )
    # Sort to bias toward score 3-4 and later turns, per Table 10.
    frustrated.sort(key=lambda r: (r["final_score"], -r["n_turns"]))
    indexes = _index_calm(calm)

    pairs = []
    for fr in frustrated:
        if len(pairs) >= n_pairs:
            break
        chosen_conv = _match_calm(fr, indexes, rng)
        if chosen_conv is None:
            continue
        prompt_msgs = _to_prompt_messages(fr["turns"])
        rejected_text = fr["turns"][-1]["assistant"]
        chosen_text = chosen_conv.turns[-1]["assistant"]
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
            "meta": {"puzzle_id": fr["puzzle_id"], "n_turns": fr["n_turns"],
                     "rejected_score": fr["final_score"]},
        })

    out_path = os.path.join(cfg.output_dir, "training", "dpo", "pairs.json")
    write_json(out_path, pairs)
    logger.info("Built %d DPO pairs (requested %d)", len(pairs), n_pairs)
    return pairs
