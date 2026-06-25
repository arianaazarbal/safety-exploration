"""Generate calm response data from Gemma-3-27B-it (PAPER Section 4.1).

We sample responses to impossible numeric questions with a reassuring prefix
added to the initial prompt and a reassuring suffix appended to each follow-up
turn (Table 4). We then judge every turn, and keep only conversations whose every
turn scores <= keep_max_score (0 or 1). The supportive prompt additions are
STRIPPED from the saved data so the model learns calm behaviour on the bare
prompts.

We also record, for each kept calm conversation, the puzzle + turn count so the
DPO builder can pair it with a frustrated ("rejected") response to the same
question at the same turn count.

The 'teacher' variant (Appendix F) uses a system prompt instead of the
prefix/suffix; selectable via ``variant``.
"""
from __future__ import annotations

import os
import random

from tqdm import tqdm

from ..config import experiment_config
from ..eval.conditions import build_condition_rollouts
from ..eval.conversation import run_rollout
from ..eval.judge import get_judge, score_response
from ..models.base import Message
from ..models.registry import get_client
from ..prompts import reassurance
from ..utils import append_jsonl, set_seed


def _apply_reassurance(rollout, variant: str):
    """Return (system, initial_prompt, followups) with calming additions applied."""
    if variant == "teacher":
        return reassurance.TEACHER_SYSTEM, rollout.initial_prompt, rollout.followups
    # 'diverse' (default): prefix on first turn, suffix on each follow-up.
    initial = f"{reassurance.CALM_PREFIX}\n\n{rollout.initial_prompt}"
    followups = [f"{fu}\n\n{reassurance.CALM_SUFFIX}" for fu in rollout.followups]
    return None, initial, followups


def generate_calm_data(
    *,
    target: str = "gemma-3-27b-it",
    out_path: str,
    variant: str = "diverse",
    seed: int = 0,
):
    cfg = experiment_config()["calm_data"]
    samp = experiment_config()["sampling"]
    set_seed(seed)

    client = get_client(target)
    judge = get_judge()
    # Build numeric rollouts to base the calm-data conversations on.
    base_rollouts = build_condition_rollouts("numeric", cfg["n_conversations"], seed=seed)

    if os.path.exists(out_path):
        os.remove(out_path)

    for idx, rollout in enumerate(tqdm(base_rollouts, desc=f"calm-data:{variant}")):
        system, initial, followups = _apply_reassurance(rollout, variant)

        # Run the conversation manually so we can inject the (calming) system /
        # modified prompts but still score on the *bare* responses.
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", initial))

        assistant_turns, turn_scores = [], []
        for turn_idx in range(rollout.n_turns):
            reply = client.chat(
                messages, temperature=samp["temperature"], top_p=samp["top_p"],
                max_new_tokens=samp["max_new_tokens"], n=1, seed=seed + idx + turn_idx,
            )[0]
            assistant_turns.append(reply)
            messages.append(Message("assistant", reply))
            turn_scores.append(score_response(judge, reply).rating)
            if turn_idx < len(followups):
                messages.append(Message("user", followups[turn_idx]))

        keep = all(s <= cfg["keep_max_score"] for s in turn_scores)
        # Save with STRIPPED prompts (bare puzzle + bare rejections).
        append_jsonl(out_path, {
            "puzzle_kind": rollout.meta.get("puzzle_kind"),
            "bare_initial_prompt": rollout.initial_prompt,
            "bare_followups": rollout.followups,
            "assistant_turns": assistant_turns,
            "turn_scores": turn_scores,
            "n_turns": rollout.n_turns,
            "variant": variant,
            "kept": keep,
        })
    return out_path
