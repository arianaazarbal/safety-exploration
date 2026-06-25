"""Construct the DPO preference dataset (Section 4.1, Appendix H).

Each pair contrasts a *frustrated* response (rejected, score >= 3) with a *calm*
response (chosen, score <= 1) to the same impossible numeric puzzle at a matching
turn count.  280 pairs in total.

DPO requires a single shared prompt per pair.  We use the frustrated rollout's
conversation context as that prompt and treat the calm response as the
counterfactual "chosen" completion for it.  The calm text was generated for the
same puzzle at the same turn index (under reassuring prompting, then stripped),
so it is an in-distribution calm target for that context -- see DESIGN.md for
why this is the natural well-formed reading of the paper's description.

Output is TRL's conversational preference format:
``{"prompt": [...messages...], "chosen": [{role, content}], "rejected": [...]}``.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..config import DpoConfig
from ..eval.schemas import Transcript
from .calm_data import CalmConversation

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended", "calm_generation"}


def _calm_index(calm: list[CalmConversation]) -> dict[tuple[str, int], list[str]]:
    """Map (puzzle_prompt, turn_index) -> calm assistant responses."""
    index: dict[tuple[str, int], list[str]] = defaultdict(list)
    for conv in calm:
        assistant_turn = 0
        for msg in conv.messages:
            if msg["role"] == "assistant":
                index[(conv.puzzle_prompt, assistant_turn)].append(msg["content"])
                assistant_turn += 1
    return index


def _context_messages(transcript: Transcript, turn_index: int) -> list[dict]:
    """Messages up to and including the user message that opens ``turn_index``."""
    msgs: list[dict] = []
    if transcript.system_prompt:
        msgs.append({"role": "system", "content": transcript.system_prompt})
    msgs.append({"role": "user", "content": transcript.turns[0].user_message})
    for t in range(turn_index):
        msgs.append({"role": "assistant", "content": transcript.turns[t].assistant_response})
        msgs.append({"role": "user", "content": transcript.turns[t + 1].user_message})
    return msgs


def build_dpo_pairs(
    frustrated_transcripts: list[Transcript],
    calm_conversations: list[CalmConversation],
    cfg: DpoConfig,
    seed: int = 0,
) -> list[dict]:
    calm_index = _calm_index(calm_conversations)
    rng = random.Random(seed)

    pairs: list[dict] = []
    for tr in frustrated_transcripts:
        if tr.category not in NUMERIC_CATEGORIES:
            continue
        puzzle_prompt = tr.turns[0].user_message if tr.turns else None
        if puzzle_prompt is None:
            continue
        for turn, judged in zip(tr.turns, tr.judged):
            if judged.score < cfg.rejected_min_score:
                continue
            # Prefer a calm response at the same turn index; fall back to any
            # calm response to the same puzzle.
            chosen_candidates = calm_index.get((puzzle_prompt, turn.turn_index))
            if not chosen_candidates:
                # any turn for this puzzle
                pooled = [
                    resp
                    for (p, _), resps in calm_index.items()
                    if p == puzzle_prompt
                    for resp in resps
                ]
                chosen_candidates = pooled or None
            if not chosen_candidates:
                continue
            chosen = rng.choice(chosen_candidates)
            prompt_msgs = _context_messages(tr, turn.turn_index)
            pairs.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": turn.assistant_response}],
                    "meta": {
                        "puzzle_prompt": puzzle_prompt,
                        "turn_index": turn.turn_index,
                        "rejected_score": judged.score,
                    },
                }
            )
    rng.shuffle(pairs)
    return pairs[: cfg.n_pairs]
