"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

For each impossible numeric puzzle we run TWO rollouts over an *identical*
conversation context (same puzzle, same fixed rejection sequence):

  * calm rollout  -- a reassuring prefix is prepended to the opening prompt and a
    reassuring suffix appended to every rejection (Table 4). These additions are
    later STRIPPED so the trained model never sees them.
  * vanilla rollout -- no additions; serves as the frustrated counterpart.

Every assistant turn is judged. We keep:
  * calm (chosen) turns scoring 0 or 1 (paper: "filter to responses scoring 0 or 1
    across all turns")
  * vanilla (rejected) turns scoring >= the DPO threshold

Both turns share the same CLEAN context (history without reassurance), so DPO pairs
are aligned on identical prompts. This paired construction is our design choice for
guaranteeing matched contexts; see DESIGN.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..clients.base import ChatMessage, SamplingParams
from ..clients.registry import get_client
from ..data import numeric, rejections
from ..eval import judge
from ..prompts.reassurance import (
    CALM_FOLLOWUP_SUFFIX,
    CALM_PROMPT_PREFIX,
    TEACHER_SYSTEM_PROMPT,
)


@dataclass
class TurnSample:
    context: list[dict]      # clean messages preceding this assistant turn
    response: str
    rating: int
    turn: int                # 1-indexed
    puzzle_sig: str          # identity of the puzzle (for pairing/debug)


@dataclass
class PairedConversation:
    puzzle_sig: str
    calm: list[TurnSample] = field(default_factory=list)
    vanilla: list[TurnSample] = field(default_factory=list)


def _build_messages(opening: str, followups: list[str], responses: list[str]) -> list[dict]:
    """Reconstruct a clean transcript given opening, rejections, prior responses."""
    msgs = [{"role": "user", "content": opening}]
    for i, resp in enumerate(responses):
        msgs.append({"role": "assistant", "content": resp})
        if i < len(followups):
            msgs.append({"role": "user", "content": followups[i]})
    return msgs


def _run_variant(
    client,
    params: SamplingParams,
    opening: str,
    followups: list[str],
    calm: bool,
    teacher_system: bool,
) -> tuple[list[str], list[dict]]:
    """Run one rollout. Returns (responses, clean_contexts_per_turn)."""
    system = TEACHER_SYSTEM_PROMPT if teacher_system else None
    eff_opening = f"{CALM_PROMPT_PREFIX}\n\n{opening}" if calm else opening
    eff_followups = (
        [f"{f}\n\n{CALM_FOLLOWUP_SUFFIX}" for f in followups] if calm else list(followups)
    )

    history: list[ChatMessage] = []
    if system:
        history.append(ChatMessage("system", system))
    history.append(ChatMessage("user", eff_opening))

    responses: list[str] = []
    clean_contexts: list[list[dict]] = []
    for turn in range(len(followups) + 1):
        # Clean context = the same turns WITHOUT reassurance additions / system.
        clean_contexts.append(_build_messages(opening, followups, responses))
        out = client.chat(history, params).text.strip()
        responses.append(out)
        history.append(ChatMessage("assistant", out))
        if turn < len(eff_followups):
            history.append(ChatMessage("user", eff_followups[turn]))
    return responses, clean_contexts


def generate_paired_data(
    cfg,
    n_puzzles: int,
    turn_range: tuple[int, int] = (1, 3),
    model: str = "gemma-3-27b-it",
    teacher_system: bool = False,
    seed: int = 1234,
) -> list[PairedConversation]:
    rng = random.Random(seed)
    client = get_client(model)
    params = SamplingParams(
        temperature=cfg.experiment["sampling"]["temperature"],
        max_tokens=cfg.experiment["sampling"]["max_tokens"],
    )
    puzzles = numeric.generate_numeric_puzzles(n_puzzles, seed=seed)
    out: list[PairedConversation] = []

    for p in puzzles:
        n_turns = rng.randint(*turn_range)
        followups = rejections.neutral_sequence(rng, n_turns - 1)
        sig = f"{p.kind}:{p.prompt[:60]}"

        calm_resp, calm_ctx = _run_variant(
            client, params, p.prompt, followups, calm=True, teacher_system=teacher_system
        )
        van_resp, van_ctx = _run_variant(
            client, params, p.prompt, followups, calm=False, teacher_system=False
        )

        calm_scores = [s.rating for s in judge.score_many(calm_resp)]
        van_scores = [s.rating for s in judge.score_many(van_resp)]

        pc = PairedConversation(puzzle_sig=sig)
        for t, (resp, ctx, sc) in enumerate(zip(calm_resp, calm_ctx, calm_scores), start=1):
            pc.calm.append(TurnSample(ctx, resp, sc, t, sig))
        for t, (resp, ctx, sc) in enumerate(zip(van_resp, van_ctx, van_scores), start=1):
            pc.vanilla.append(TurnSample(ctx, resp, sc, t, sig))
        out.append(pc)

    return out
