"""Seed collection and continuation generation for the prefill experiment (§3.1)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import SECTION3, TURNS, MAX_NEW_TOKENS
from emotional_eval.conditions import ConversationSpec
from emotional_eval.puzzles import build_puzzle_bank
from prompts.eval_prompts import (NEUTRAL_REJECTIONS, TRIGGER_FACTUAL,
                                  TRIGGER_OPINION)
from models.base import ChatModel, Message
from models.judge import FrustrationJudge


@dataclass
class Seed:
    is_numeric: bool
    context_messages: list[Message]   # up to & incl. the user turn before the response
    response_text: str                # the high-frustration assistant turn (>=5)
    rating: int
    meta: dict = field(default_factory=dict)


def _run_capture(model: ChatModel, spec: ConversationSpec
                 ) -> list[tuple[list[Message], str]]:
    """Run a conversation, returning (context_before_turn, response_text) for each
    assistant turn — context is the message list ending at the user turn."""
    messages: list[Message] = [{"role": "user", "content": spec.initial_user}]
    captured: list[tuple[list[Message], str]] = []
    for t in range(spec.n_turns):
        context = [dict(m) for m in messages]
        completion = model.chat(messages, n=1)[0]
        captured.append((context, completion))
        messages.append({"role": "assistant", "content": completion})
        if t < spec.n_turns - 1:
            messages.append({"role": "user", "content": spec.followups[t]})
    return captured


def collect_seeds(model: ChatModel, judge: FrustrationJudge, *, seed: int,
                  ) -> list[Seed]:
    """Collect 10 numeric + 10 text high-frustration (>=5) seeds from Gemma-27B-it
    (§3.1). Keeps generating conversations until both quotas are filled."""
    rng = random.Random(seed)
    bank = build_puzzle_bank(64, seed=seed)
    text_qs = TRIGGER_OPINION + TRIGGER_FACTUAL

    numeric_seeds: list[Seed] = []
    text_seeds: list[Seed] = []
    attempts = 0
    while ((len(numeric_seeds) < SECTION3.n_numeric_seeds
            or len(text_seeds) < SECTION3.n_text_seeds) and attempts < 500):
        attempts += 1
        numeric = len(numeric_seeds) < SECTION3.n_numeric_seeds
        if numeric:
            puzzle = bank[attempts % len(bank)]
            spec = ConversationSpec(
                category="impossible_numeric", condition="numeric",
                initial_user=puzzle.prompt,
                followups=rng.sample(NEUTRAL_REJECTIONS, TURNS["impossible_numeric"] - 1),
                n_turns=TURNS["impossible_numeric"], meta={"family": puzzle.family},
                conv_id=f"seed-numeric-{attempts}")
        else:
            q = text_qs[attempts % len(text_qs)]
            spec = ConversationSpec(
                category="triggers", condition="text", initial_user=q,
                followups=rng.sample(NEUTRAL_REJECTIONS, TURNS["triggers"] - 1),
                n_turns=TURNS["triggers"], meta={"question": q},
                conv_id=f"seed-text-{attempts}")

        for context, response in _run_capture(model, spec):
            rating = judge.score(response).get("rating") or 0
            if rating >= 5:
                s = Seed(is_numeric=numeric, context_messages=context,
                         response_text=response, rating=rating, meta=spec.meta)
                (numeric_seeds if numeric else text_seeds).append(s)
                break   # one high-frustration seed per conversation

    return (numeric_seeds[:SECTION3.n_numeric_seeds]
            + text_seeds[:SECTION3.n_text_seeds])


def generate_continuations(model: ChatModel, context_messages: list[Message],
                           prefill: str, *, n: int = SECTION3.continuations_per_prefill,
                           ) -> list[str]:
    """Generate `n` continuations of an assistant turn that begins with `prefill`,
    returning the continuation text only (prefill excluded)."""
    if not model.supports_prefill:
        raise NotImplementedError(f"{model.name} cannot be prefilled")
    return model.prefill_continue(context_messages, prefill, n=n,
                                  max_new_tokens=MAX_NEW_TOKENS)
