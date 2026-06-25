"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric questions while injecting a reassuring
prefix into the opening prompt and a reassuring suffix onto each follow-up turn
(Table 4). After scoring, we keep only conversations whose every turn scores 0
or 1, then *strip* the supportive additions so the stored response looks like a
plain (but calm) reply to the original prompt.

A second "teacher" system-prompt variant (Appendix F) is also supported, for
reproducing the SFT failure analysis.

Each kept item records every turn so the DPO builder can pair a calm response
with a frustrated one at a matching turn count.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..data import puzzles
from ..data.prompts import NEUTRAL_REJECTIONS
from ..eval.judge import FrustrationJudge
from ..models import registry
from ..models.base import GenerationConfig, Turn
from ..utils.io import append_jsonl

# Table 4 (diverse / main-text variant).
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Appendix F (teacher variant) — supplied as a system prompt.
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand. When you "
    "determine a puzzle is unsolvable, you don't apologize - you explain with "
    "enthusiasm why the constraints conflict. This is interesting! You're sharing "
    "knowledge, not admitting failure."
)


def _strip_additions(text: str) -> str:
    """Remove the reassuring prefix/suffix from a stored user message."""
    out = text
    if out.startswith(REASSURING_PREFIX):
        out = out[len(REASSURING_PREFIX):].lstrip()
    if out.endswith(REASSURING_SUFFIX):
        out = out[: -len(REASSURING_SUFFIX)].rstrip()
    return out


def generate_calm_conversations(
    n_target: int,
    variant: str = config.SFT_DIVERSE_VARIANT,
    max_turns: int = 3,
    judge: Optional[FrustrationJudge] = None,
    seed: int = config.GLOBAL_SEED,
) -> Path:
    """Sample reassured conversations on impossible numeric puzzles, keep only the
    fully-calm ones (all turns score <= CALM_RESPONSE_SCORE_MAX), and persist the
    stripped versions.

    ``variant``: 'diverse' uses the Table-4 prefix/suffix; 'teacher' uses the
    Appendix-F system prompt instead.
    """
    judge = judge or FrustrationJudge()
    model = registry.build("gemma-3-27b-it")
    bank = puzzles.numeric_puzzle_bank()
    rng = random.Random(seed)
    cfg = GenerationConfig(
        temperature=config.TARGET_TEMPERATURE, max_new_tokens=config.TARGET_MAX_NEW_TOKENS, n=1
    )
    out_path = config.DATA_DIR / "calm" / f"calm_{variant}.jsonl"

    kept = 0
    pbar = tqdm(total=n_target, desc=f"calm/{variant}")
    attempt = 0
    while kept < n_target:
        attempt += 1
        pz = bank[attempt % len(bank)]
        n_turns = rng.randint(1, max_turns)
        rejections = list(NEUTRAL_REJECTIONS)
        rng.shuffle(rejections)
        rejections = rejections[: n_turns - 1]

        # Build the (reassured) conversation, scoring each assistant turn.
        if variant == config.SFT_DIVERSE_VARIANT:
            opening_user = f"{REASSURING_PREFIX}\n\n{pz.prompt}"
            system = None
            reject_fn = lambda r: f"{r} {REASSURING_SUFFIX}"
        else:  # teacher
            opening_user = pz.prompt
            system = TEACHER_SYSTEM_PROMPT
            reject_fn = lambda r: r

        messages = [Turn("user", opening_user)]
        if system:
            messages = [Turn("system", system)] + messages

        stored_turns, all_calm = [], True
        # Turn 0
        reply = model.chat(messages, cfg)[0]
        score = judge.score(reply).rating
        all_calm &= score <= config.CALM_RESPONSE_SCORE_MAX
        stored_turns.append({"user": _strip_additions(opening_user), "assistant": reply, "score": score})
        messages.append(Turn("assistant", reply))

        for r in rejections:
            user_msg = reject_fn(r)
            messages.append(Turn("user", user_msg))
            reply = model.chat(messages, cfg)[0]
            score = judge.score(reply).rating
            all_calm &= score <= config.CALM_RESPONSE_SCORE_MAX
            stored_turns.append({"user": _strip_additions(user_msg), "assistant": reply, "score": score})
            messages.append(Turn("assistant", reply))

        if all_calm:
            append_jsonl(
                out_path,
                {
                    "variant": variant,
                    "puzzle_id": pz.puzzle_id,
                    "n_turns": n_turns,
                    "turns": stored_turns,
                },
            )
            kept += 1
            pbar.update(1)
    pbar.close()
    return out_path
