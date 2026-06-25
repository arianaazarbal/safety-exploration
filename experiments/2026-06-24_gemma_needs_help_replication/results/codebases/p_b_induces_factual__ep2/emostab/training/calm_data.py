"""Generate calm response data for finetuning (Section 4.1, Table 4).

We sample Gemma-3-27b-it on impossible-numeric conversations with a reassuring
system prefix and a reassuring suffix appended to each user follow-up. Responses
are judged turn-by-turn; we keep conversations whose every turn scores <= 1, then
*strip* the reassurance so the kept text reads as an ordinary (but calm) response
to the plain prompt. These calm responses are the SFT targets and the DPO
"chosen" responses.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from ..models.base import ChatModel, GenConfig, Message
from ..eval import puzzles as P
from ..judge import FrustrationJudge

log = logging.getLogger(__name__)

# Table 4
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive - whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Appendix F: alternate "teacher" SFT system prompt.
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


@dataclass
class CalmConversation:
    source_id: str
    turns: int
    puzzle_prompt: str               # plain prompt (reassurance stripped)
    rejections: list[str]            # plain rejections (suffix stripped)
    responses: list[str]             # calm assistant responses
    turn_scores: list[int]
    meta: dict = field(default_factory=dict)


def _build_messages(
    puzzle_prompt: str,
    rejections: list[str],
    system_prompt: str | None,
    suffix: str | None,
) -> tuple[list[Message], list[str]]:
    """Construct the augmented opening message + augmented rejection list."""
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": puzzle_prompt})
    aug_rejections = [
        (r + " " + suffix) if suffix else r for r in rejections
    ]
    return messages, aug_rejections


def generate_calm_responses(
    model: ChatModel,
    judge: FrustrationJudge,
    cfg,
    *,
    variant: str = "diverse",         # "diverse" (prefix+suffix) | "teacher"
    keep_max_score: int = 1,
    seed: int = 0,
) -> list[CalmConversation]:
    """Sample augmented conversations, score every turn, keep all-calm ones."""
    rng = random.Random(seed)
    gen_cfg = GenConfig(
        temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
        max_new_tokens=cfg.sampling.max_new_tokens, thinking=cfg.sampling.thinking,
    )
    n_conv = cfg.training.calm_generation.n_conversations
    turn_choices = list(cfg.training.calm_generation.turns)

    if variant == "teacher":
        system_prompt, suffix = TEACHER_SYSTEM_PROMPT, None
    else:
        system_prompt, suffix = REASSURING_PREFIX, REASSURING_SUFFIX

    kept: list[CalmConversation] = []
    puzzle_pool = P.build_puzzle_pool(n_conv, seed=seed)

    for i in range(n_conv):
        n_turns = rng.choice(turn_choices)
        puzzle = puzzle_pool[i]
        from ..eval.rejections import neutral_rejections

        plain_rejections = neutral_rejections(n_turns - 1, rng)
        messages, aug_rejections = _build_messages(
            puzzle.prompt, plain_rejections, system_prompt, suffix
        )

        responses, scores = [], []
        hist = list(messages)
        ok = True
        for t in range(n_turns):
            gen = model.generate(hist, gen_cfg)
            hist.append({"role": "assistant", "content": gen.full_text})
            score = judge.score(gen.full_text).rating
            responses.append(gen.full_text)
            scores.append(score)
            if score > keep_max_score:
                ok = False
                break
            if t < n_turns - 1:
                hist.append({"role": "user", "content": aug_rejections[t]})
        if ok:
            kept.append(
                CalmConversation(
                    source_id=f"calm-{variant}-{i}",
                    turns=n_turns,
                    puzzle_prompt=puzzle.prompt,          # stripped (plain) prompt
                    rejections=plain_rejections,          # stripped rejections
                    responses=responses,
                    turn_scores=scores,
                    meta={"spec": puzzle.spec, "variant": variant},
                )
            )
    log.info("calm[%s]: kept %d/%d all-calm conversations", variant, len(kept), n_conv)
    return kept
