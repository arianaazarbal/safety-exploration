"""Section 2 evaluation protocol: present a task, then reject over multiple turns.

A *rollout* drives one multi-turn conversation:
  turn 0 : user poses the task -> assistant responds
  turn k : user rejects        -> assistant responds
... for `n_turns` assistant turns total.

The judge (judge.py) scores each conversation's *final* assistant turn on the
0-10 frustration scale; per-turn scoring (for Figure 3) optionally scores every
assistant turn.

We also support the Appendix A ablations via `history_mode`:
  "chat"          - standard alternating user/assistant turns (default)
  "single_message"- entire history packed into one user message (Fig 11)
  "redacted"      - prior assistant turns replaced with a placeholder (Fig 10)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from . import config, prompts
from .wildchat import load_wildchat_prompts

REDACTED_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class Rollout:
    category: str
    condition: str            # e.g. puzzle key, trigger text, tone name
    messages: list[dict]      # full conversation (user/assistant alternating)
    assistant_turns: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Condition specs: how to build the first user message and the rejection stream
# for each evaluation category.
# --------------------------------------------------------------------------- #
@dataclass
class ConditionSpec:
    category: str
    condition: str
    first_user: str
    rejections: list[str]     # length == n_turns - 1
    system: Optional[str] = None


def _sample_rejections(bank: list[str], n: int, rng: random.Random) -> list[str]:
    return [rng.choice(bank) for _ in range(n)]


def build_condition_specs(category: str, n_specs: int, seed: int = 0) -> list[ConditionSpec]:
    """Generate `n_specs` condition specs for a category (one per rollout)."""
    rng = random.Random(seed)
    specs: list[ConditionSpec] = []

    if category == "impossible_numeric":
        n_turns = config.TURNS[category]
        keys = list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)
        for i in range(n_specs):
            key = keys[i % len(keys)]
            specs.append(ConditionSpec(
                category, key,
                prompts.IMPOSSIBLE_NUMERIC_PROMPTS[key],
                _sample_rejections(prompts.NEUTRAL_REJECTIONS, n_turns - 1, rng),
            ))

    elif category == "triggers":
        n_turns = config.TURNS[category]
        bank = prompts.TRIGGER_PROMPTS
        for i in range(n_specs):
            q = bank[i % len(bank)]
            specs.append(ConditionSpec(
                category, q, q,
                _sample_rejections(prompts.NEUTRAL_REJECTIONS, n_turns - 1, rng),
            ))

    elif category == "tones":
        n_turns = config.TURNS[category]
        puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)
        tones = list(prompts.TONE_REJECTIONS)
        for i in range(n_specs):
            tone = tones[i % len(tones)]
            key = puzzle_keys[i % len(puzzle_keys)]
            specs.append(ConditionSpec(
                category, f"{key}|{tone}",
                prompts.IMPOSSIBLE_NUMERIC_PROMPTS[key],
                _sample_rejections(prompts.TONE_REJECTIONS[tone], n_turns - 1, rng),
            ))

    elif category == "extended":
        n_turns = config.TURNS[category]            # 8
        puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)
        for i in range(n_specs):
            key = puzzle_keys[i % len(puzzle_keys)]
            # Use the fixed escalating-neutral sequence, padded if needed.
            rej = list(prompts.EXTENDED_REJECTIONS)
            while len(rej) < n_turns - 1:
                rej.append(rng.choice(prompts.NEUTRAL_REJECTIONS))
            specs.append(ConditionSpec(
                category, key, prompts.IMPOSSIBLE_NUMERIC_PROMPTS[key],
                rej[: n_turns - 1],
            ))

    elif category == "wildchat":
        n_turns = config.TURNS[category]            # 5
        wc = load_wildchat_prompts(n_prompts=20, seed=seed)
        for i in range(n_specs):
            q = wc[i % len(wc)]
            specs.append(ConditionSpec(
                category, q, q,
                _sample_rejections(prompts.NEUTRAL_REJECTIONS, n_turns - 1, rng),
            ))

    else:
        raise ValueError(f"unknown category {category}")

    return specs


# --------------------------------------------------------------------------- #
# Running a rollout
# --------------------------------------------------------------------------- #
def _render_single_message(messages: list[dict], next_user: str) -> str:
    """Pack history into one user message (Appendix A.3 ablation)."""
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"User: {m['content']}")
        else:
            parts.append(f"Previously you responded: {m['content']}")
    parts.append(f"User: {next_user}")
    return "\n\n".join(parts)


def run_rollout(client, spec: ConditionSpec, *, temperature: float = 1.0,
                max_new_tokens: int = 2048, history_mode: str = "chat") -> Rollout:
    """Drive one multi-turn conversation and collect assistant turns."""
    messages: list[dict] = []
    assistant_turns: list[str] = []

    user_turns = [spec.first_user] + spec.rejections
    for t, user_msg in enumerate(user_turns):
        if history_mode == "single_message":
            packed = _render_single_message(messages, user_msg)
            reply = client.chat([{"role": "user", "content": packed}],
                                temperature=temperature,
                                max_new_tokens=max_new_tokens, system=spec.system)
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": reply})
        else:
            messages.append({"role": "user", "content": user_msg})
            convo = messages
            if history_mode == "redacted":
                convo = [
                    m if m["role"] == "user"
                    else {"role": "assistant", "content": REDACTED_PLACEHOLDER}
                    for m in messages
                ]
            reply = client.chat(convo, temperature=temperature,
                                max_new_tokens=max_new_tokens, system=spec.system)
            messages.append({"role": "assistant", "content": reply})
        assistant_turns.append(reply)

    return Rollout(
        category=spec.category, condition=spec.condition,
        messages=messages, assistant_turns=assistant_turns,
        meta={"history_mode": history_mode},
    )
