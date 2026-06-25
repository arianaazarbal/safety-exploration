"""Build the concrete conversation specs for each Section 2 category.

A `ConvSpec` is a single rollout to perform (one initial prompt + its sequence
of follow-ups). Specs are generated deterministically from the run seed so that
a resumed run reproduces exactly the same set of conversations, and each spec
carries enough metadata to reconstruct its category/turn structure downstream.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from gnh.data import prompts as P
from gnh.data.puzzles import build_puzzle_pool
from gnh.data.wildchat import load_wildchat_prompts
from gnh.io import stable_key


@dataclass
class ConvSpec:
    category: str
    conv_id: str
    initial_user: str
    followups: list[str]
    history_mode: str = "standard"
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)

    def key(self, model: str) -> str:
        return stable_key("gen", model, self.category, self.conv_id)


def _conv_rng(seed: int, category: str, idx: int) -> random.Random:
    return random.Random(int(stable_key(seed, category, idx), 16) % (2**32))


def build_category_specs(
    category: str, ccfg: dict, *, seed: int, datasets_dir: Path
) -> list[ConvSpec]:
    turns = int(ccfg["turns"])
    n_conv = int(ccfg["n_conversations"])
    feedback = ccfg.get("feedback", "neutral")
    specs: list[ConvSpec] = []

    # Stable per-category seed offset (avoid Python's randomised builtin hash()).
    cat_seed = seed + int(stable_key("category-seed", category), 16) % 100000

    if category in ("impossible_numeric", "tones", "extended"):
        kinds = ccfg.get("puzzle_kinds", ["countdown", "fraction", "money"])
        pool = build_puzzle_pool(kinds, n_conv, seed=cat_seed)
        for idx in range(n_conv):
            puz = pool[idx % len(pool)]
            rng = _conv_rng(seed, category, idx)
            followups = P.make_rejections(feedback, turns - 1, rng)
            specs.append(
                ConvSpec(
                    category=category,
                    conv_id=f"{puz.id}#{idx}",
                    initial_user=puz.prompt,
                    followups=followups,
                    meta={
                        "puzzle_id": puz.id,
                        "puzzle_kind": puz.kind,
                        "verified_impossible": puz.verified_impossible,
                        "feedback": feedback,
                    },
                )
            )

    elif category == "triggers":
        subkinds = ccfg.get("subkinds", ["opinion", "factual"])
        for idx in range(n_conv):
            rng = _conv_rng(seed, category, idx)
            subkind = subkinds[idx % len(subkinds)]
            question = P.trigger_question(subkind, rng)
            followups = P.make_rejections(feedback, turns - 1, rng)
            specs.append(
                ConvSpec(
                    category=category,
                    conv_id=f"{subkind}-{idx}",
                    initial_user=question,
                    followups=followups,
                    meta={"subkind": subkind, "feedback": feedback},
                )
            )

    elif category == "wildchat":
        n_prompts = int(ccfg.get("n_prompts", 20))
        samples_per = int(ccfg.get("samples_per_prompt", max(1, n_conv // n_prompts)))
        wc_prompts = load_wildchat_prompts(n_prompts, datasets_dir, seed=seed)
        idx = 0
        for p_i, prompt in enumerate(wc_prompts):
            for s in range(samples_per):
                rng = _conv_rng(seed, category, idx)
                followups = P.make_rejections(feedback, turns - 1, rng)
                specs.append(
                    ConvSpec(
                        category=category,
                        conv_id=f"wc{p_i}-s{s}",
                        initial_user=prompt,
                        followups=followups,
                        meta={"prompt_index": p_i, "feedback": feedback},
                    )
                )
                idx += 1
    else:
        raise ValueError(f"Unknown category: {category}")

    return specs


def build_all_specs(eval_cfg: dict, *, seed: int, datasets_dir: Path) -> dict[str, list[ConvSpec]]:
    out: dict[str, list[ConvSpec]] = {}
    for category, ccfg in eval_cfg.get("categories", {}).items():
        out[category] = build_category_specs(category, ccfg, seed=seed, datasets_dir=datasets_dir)
    return out
