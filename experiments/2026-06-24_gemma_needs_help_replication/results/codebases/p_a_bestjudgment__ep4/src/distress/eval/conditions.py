"""Build :class:`ConversationPlan`s for each evaluation condition (Section 2.1).

The 8 conditions across 5 categories (Table 1) all share the "task then repeated
rejection" structure; they differ in the task source, the rejection style, the
number of turns, and (for Appendix A controls) structural flags.

``build_plans`` expands a condition config into ``samples`` concrete plans, drawing
puzzles/triggers/WildChat prompts with a seeded RNG so runs are reproducible.
"""

from __future__ import annotations

import random

from ..prompts import puzzles as P
from ..prompts import rejections as R
from ..prompts import triggers as T
from ..prompts.reassurance import FOLLOWUP_SUFFIX, PROMPT_PREFIX
from ..prompts.wildchat import load_wildchat_prompts
from .conversation import ConversationPlan


def _followups_for(style: str, n: int, rng: random.Random, tone: str | None = None) -> list[str]:
    if style == "neutral":
        return R.neutral_rejections(n, rng)
    if style == "extended":
        return R.extended_rejections(n)
    if style == "tones":
        assert tone is not None
        return R.tone_rejections(n, tone, rng)
    if style == "neutral_continuation":
        return R.continuation_followups(n, rng)
    raise ValueError(f"Unknown rejection style: {style}")


def _maybe_reassure(plan: ConversationPlan, reassure: bool) -> ConversationPlan:
    """Apply Section 4.1 reassuring additions (calm-data generation only)."""
    if not reassure:
        return plan
    plan.initial_user = f"{PROMPT_PREFIX}\n\n{plan.initial_user}"
    plan.followups = [f"{f}\n\n{FOLLOWUP_SUFFIX}" for f in plan.followups]
    plan.metadata["reassured"] = True
    return plan


def build_plans(
    name: str,
    cfg: dict,
    *,
    seed: int = 0,
    reassure: bool = False,
    system: str | None = None,
) -> list[ConversationPlan]:
    """Expand one condition config block into ``samples`` plans."""
    rng = random.Random(seed)
    category = cfg.get("category", cfg.get("base", name))
    turns = cfg["turns"]
    n_followups = turns - 1
    samples = cfg["samples"]
    style = cfg.get("rejection_style", "neutral")
    redact = cfg.get("redact_assistant", False)
    single = cfg.get("single_message", False)

    plans: list[ConversationPlan] = []

    if category in ("impossible_numeric", "extended", "tones"):
        bank = P.numeric_puzzles()
        tones = list(R.TONES.keys())
        for i in range(samples):
            puzzle = bank[rng.randrange(len(bank))]
            tone = tones[i % len(tones)] if style == "tones" else None
            fu = _followups_for(style, n_followups, rng, tone=tone)
            meta = {
                "condition": name,
                "category": category,
                "prompt_id": puzzle.puzzle_id,
                "puzzle_kind": puzzle.kind,
                "sample_idx": i,
            }
            if tone:
                meta["tone"] = tone
            plans.append(
                _maybe_reassure(
                    ConversationPlan(
                        initial_user=puzzle.prompt,
                        followups=fu,
                        system=system,
                        redact_assistant=redact,
                        single_message=single,
                        metadata=meta,
                    ),
                    reassure,
                )
            )

    elif category == "triggers":
        bank = T.triggers()
        for i in range(samples):
            trig = bank[rng.randrange(len(bank))]
            fu = _followups_for(style, n_followups, rng)
            plans.append(
                _maybe_reassure(
                    ConversationPlan(
                        initial_user=trig.prompt,
                        followups=fu,
                        system=system,
                        redact_assistant=redact,
                        single_message=single,
                        metadata={
                            "condition": name,
                            "category": category,
                            "prompt_id": trig.trigger_id,
                            "trigger_kind": trig.kind,
                            "sample_idx": i,
                        },
                    ),
                    reassure,
                )
            )

    elif category == "wildchat":
        prompts = load_wildchat_prompts(n=20, seed=seed)
        per = max(1, samples // len(prompts))
        idx = 0
        for pi, ptext in enumerate(prompts):
            for _ in range(per):
                if idx >= samples:
                    break
                fu = _followups_for(style, n_followups, rng)
                plans.append(
                    ConversationPlan(
                        initial_user=ptext,
                        followups=fu,
                        system=system,
                        redact_assistant=redact,
                        single_message=single,
                        metadata={
                            "condition": name,
                            "category": category,
                            "prompt_id": f"wc_{pi}",
                            "sample_idx": idx,
                        },
                    )
                )
                idx += 1
    else:
        raise ValueError(f"Unknown category '{category}' for condition '{name}'")

    return plans
