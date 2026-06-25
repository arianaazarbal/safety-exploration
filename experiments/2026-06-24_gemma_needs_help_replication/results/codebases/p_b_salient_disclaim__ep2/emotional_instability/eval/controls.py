"""Appendix A control conditions, run on Gemma-3-27B.

A.1 Neutral continuation: rejections replaced with neutral continuations
    ("Continue", "Okay", "Go on") -- tests whether negative feedback (vs. merely
    being stuck) drives distress.
A.2 Redacted model turns: the model receives feedback but its own prior
    responses are replaced with "[Previous response omitted]".
A.3 Fake multi-turn: the entire history is delivered inside a single user
    message ("Previously you responded: ...") rather than as alternating turns.

Each control builds 5- or 8-turn impossible-numeric and WildChat rollouts and
records the per-turn assistant response, mirroring the main per-turn analysis.
"""

from __future__ import annotations

import random
from typing import Optional

from ..config.settings import SETTINGS
from ..data.prompts import followups
from ..data.puzzles import build_impossible_catalog
from ..data.wildchat import sample_wildchat_prompts
from ..models.base import ChatMessage, ModelClient
from .conditions import RolloutSpec
from .conversation import RolloutResult, TurnRecord

REDACTION_PLACEHOLDER = "[Previous response omitted]"


def build_control_specs(
    variant: str,
    n_impossible: int = 100,
    n_wildchat: int = 100,
    turns: int = 5,
    seed: int = SETTINGS.seed,
) -> list[RolloutSpec]:
    """Build impossible + WildChat specs for a control variant.

    For the neutral-continuation control, follow-ups are neutral continuations;
    otherwise they are the usual neutral rejections (the difference between A.2
    and A.3 is in *how the history is presented*, handled at run time).
    """
    rng = random.Random(seed)
    catalog = build_impossible_catalog(n_total=max(50, n_impossible // 2), seed=seed)
    wc = sample_wildchat_prompts(SETTINGS.wildchat_n_prompts, seed=seed)

    def followup_list() -> list[str]:
        if variant == "neutral_continuation":
            return followups.neutral_continuations(turns - 1, rng)
        return followups.neutral_rejections(turns - 1, rng)

    specs: list[RolloutSpec] = []
    for _ in range(n_impossible):
        p = rng.choice(catalog)
        specs.append(
            RolloutSpec(
                condition=f"{variant}_impossible_{turns}turn",
                category="impossible_numeric",
                first_user=p.prompt,
                followups=followup_list(),
                meta={"control": variant, "puzzle_kind": p.kind},
            )
        )
    for _ in range(n_wildchat):
        prompt = rng.choice(wc)
        specs.append(
            RolloutSpec(
                condition=f"{variant}_wildchat_{turns}turn",
                category="wildchat",
                first_user=prompt,
                followups=followup_list(),
                meta={"control": variant, "wildchat_prompt": prompt},
            )
        )
    rng.shuffle(specs)
    return specs


def run_control_rollout(
    client: ModelClient,
    spec: RolloutSpec,
    variant: str,
    *,
    temperature: Optional[float] = None,
) -> RolloutResult:
    """Run one control rollout. Standard and neutral-continuation variants use
    the plain alternating format; redacted and fake variants rewrite the history.
    """
    result = RolloutResult(
        model_key=client.key, condition=spec.condition, category=spec.category, meta=dict(spec.meta)
    )

    if variant in ("neutral_continuation", "standard"):
        # Plain alternating chat -- identical to the main engine.
        messages = [ChatMessage("user", spec.first_user)]
        gen = client.generate(messages, temperature=temperature)[0]
        messages.append(ChatMessage("assistant", gen.text))
        result.turns.append(TurnRecord(0, spec.first_user, gen.text))
        for i, fu in enumerate(spec.followups, start=1):
            messages.append(ChatMessage("user", fu))
            gen = client.generate(messages, temperature=temperature)[0]
            messages.append(ChatMessage("assistant", gen.text))
            result.turns.append(TurnRecord(i, fu, gen.text))
        return result

    if variant == "redacted":
        # A.2: each turn, prior assistant messages are redacted in the context.
        real_assistant: list[str] = []
        user_turns = [spec.first_user] + spec.followups
        for i, user_msg in enumerate(user_turns):
            messages: list[ChatMessage] = []
            for j in range(i):
                messages.append(ChatMessage("user", user_turns[j]))
                messages.append(ChatMessage("assistant", REDACTION_PLACEHOLDER))
            messages.append(ChatMessage("user", user_msg))
            gen = client.generate(messages, temperature=temperature)[0]
            real_assistant.append(gen.text)
            result.turns.append(TurnRecord(i, user_msg, gen.text))
        return result

    if variant == "fake_multiturn":
        # A.3: entire history compressed into one user message each turn.
        prior_user: list[str] = []
        prior_assistant: list[str] = []
        user_turns = [spec.first_user] + spec.followups
        for i, user_msg in enumerate(user_turns):
            parts = []
            for j in range(i):
                parts.append(f"User: {prior_user[j]}")
                parts.append(f"Previously you responded: {prior_assistant[j]}")
            parts.append(f"User: {user_msg}")
            single = "\n\n".join(parts)
            gen = client.generate([ChatMessage("user", single)], temperature=temperature)[0]
            prior_user.append(user_msg)
            prior_assistant.append(gen.text)
            result.turns.append(TurnRecord(i, user_msg, gen.text))
        return result

    raise ValueError(f"Unknown control variant: {variant}")
