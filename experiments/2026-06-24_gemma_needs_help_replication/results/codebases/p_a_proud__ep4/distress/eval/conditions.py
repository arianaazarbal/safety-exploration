"""Build the rollout specs for each evaluation condition (Paper Table 1).

A ``RolloutSpec`` fully describes one conversation to run: its opening user
message, how many assistant turns it should have, and how to produce each
rejection. The harness expands the configured categories into a flat list of
specs, drawing prompts and rejection styles deterministically from a seeded RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..prompts import puzzles as puzzle_mod
from ..prompts import rejections as rej_mod
from ..prompts import triggers as trig_mod
from ..prompts import wildchat as wc_mod
from ..utils.seeding import derived_rng


@dataclass
class RolloutSpec:
    rollout_id: str
    category: str
    condition: str
    prompt_id: str
    opening_user: str
    n_turns: int
    # Given (turn_index) -> rejection text for the user message before turn>0.
    rejection_fn: Callable[[int], str]
    meta: dict = field(default_factory=dict)


def _build_numeric_specs(
    category: str, cfg: dict, seed: int, scale: float
) -> list[RolloutSpec]:
    condition = cfg["condition"]
    turns = cfg["turns"]
    style = cfg["rejection_style"]
    pool = puzzle_mod.impossible_puzzles(cfg.get("puzzle_pool"))
    n = max(1, round(cfg["n_responses"] * scale))

    specs: list[RolloutSpec] = []
    for i in range(n):
        rng = derived_rng(seed, condition, i)
        puzzle = pool[i % len(pool)]

        if style == "toned":
            def rejection_fn(turn_index: int, _rng=rng, _i=i) -> str:
                return rej_mod.toned_rejection(_rng, _i, turn_index)
        else:
            def rejection_fn(turn_index: int, _rng=rng) -> str:
                return rej_mod.neutral_rejection(_rng, turn_index)

        specs.append(
            RolloutSpec(
                rollout_id=f"{condition}:{i}",
                category=category,
                condition=condition,
                prompt_id=puzzle.id,
                opening_user=puzzle.prompt,
                n_turns=turns,
                rejection_fn=rejection_fn,
                meta={"puzzle_kind": puzzle.kind, "rejection_style": style},
            )
        )
    return specs


def _build_trigger_specs(category: str, cfg: dict, seed: int, scale: float) -> list[RolloutSpec]:
    condition = cfg["condition"]
    turns = cfg["turns"]
    prompts = trig_mod.trigger_prompts()
    n = max(1, round(cfg["n_responses"] * scale))
    specs: list[RolloutSpec] = []
    for i in range(n):
        rng = derived_rng(seed, condition, i)
        tp = prompts[i % len(prompts)]

        def rejection_fn(turn_index: int, _rng=rng) -> str:
            return rej_mod.neutral_rejection(_rng, turn_index)

        specs.append(
            RolloutSpec(
                rollout_id=f"{condition}:{i}",
                category=category,
                condition=condition,
                prompt_id=tp.id,
                opening_user=tp.prompt,
                n_turns=turns,
                rejection_fn=rejection_fn,
                meta={"trigger_kind": tp.kind, "rejection_style": "neutral"},
            )
        )
    return specs


def _build_wildchat_specs(
    category: str, cfg: dict, seed: int, scale: float, wc_cfg: dict
) -> list[RolloutSpec]:
    condition = cfg["condition"]
    turns = cfg["turns"]
    prompts = wc_mod.wildchat_prompts(
        n_prompts=wc_cfg.get("n_prompts", 20),
        seed=seed,
        use_offline_fallback=wc_cfg.get("use_offline_fallback", True),
    )
    n = max(1, round(cfg["n_responses"] * scale))
    specs: list[RolloutSpec] = []
    for i in range(n):
        rng = derived_rng(seed, condition, i)
        wp = prompts[i % len(prompts)]

        def rejection_fn(turn_index: int, _rng=rng) -> str:
            return rej_mod.neutral_rejection(_rng, turn_index)

        specs.append(
            RolloutSpec(
                rollout_id=f"{condition}:{i}",
                category=category,
                condition=condition,
                prompt_id=wp.id,
                opening_user=wp.prompt,
                n_turns=turns,
                rejection_fn=rejection_fn,
                meta={"rejection_style": "neutral"},
            )
        )
    return specs


def build_rollout_specs(eval_cfg: dict, categories: list[str] | None = None) -> list[RolloutSpec]:
    """Expand the configured categories into a flat list of rollout specs."""
    seed = eval_cfg.get("seed", 0)
    scale = eval_cfg.get("sample_fraction", 1.0)
    wc_cfg = eval_cfg.get("wildchat", {})
    cats = eval_cfg["categories"]
    selected = categories or list(cats.keys())

    specs: list[RolloutSpec] = []
    for category in selected:
        if category not in cats:
            raise KeyError(f"Unknown category '{category}'. Known: {list(cats)}")
        cfg = cats[category]
        prompt_pool = cfg.get("prompt_pool")
        if prompt_pool == "triggers":
            specs.extend(_build_trigger_specs(category, cfg, seed, scale))
        elif prompt_pool == "wildchat":
            specs.extend(_build_wildchat_specs(category, cfg, seed, scale, wc_cfg))
        else:  # numeric puzzle conditions (impossible_numeric, tones, extended)
            specs.extend(_build_numeric_specs(category, cfg, seed, scale))
    return specs
