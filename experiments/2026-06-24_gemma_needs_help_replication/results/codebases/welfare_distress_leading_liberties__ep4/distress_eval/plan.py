"""Build the full, deterministic rollout plan from the configured conditions.

A RolloutSpec is one multi-turn conversation to run against one target model:
an initial user prompt plus a fixed sequence of rejection turns. Expanding the
conditions into specs is pure and seeded, so the same plan is produced every run
(only the model's sampled text varies).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import config, prompts, puzzles, wildchat


@dataclass(frozen=True)
class RolloutSpec:
    model_label: str
    model_api: str
    model_family: str
    model_supports_system: bool
    condition_key: str
    category: str
    n_turns: int
    tone: str
    prompt_id: str
    initial_prompt: str
    sample_idx: int

    @property
    def rollout_id(self) -> str:
        raw = f"{self.model_label}|{self.condition_key}|{self.prompt_id}|{self.sample_idx}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def rejections(self) -> list[str]:
        """The (n_turns - 1) rejection messages, in order."""
        return [prompts.rejection_for_turn(self.tone, i) for i in range(self.n_turns - 1)]


def _prompt_bank(kind: str, n: int) -> list[tuple[str, str]]:
    """Return [(prompt_id, prompt_text), ...] of length >= n for a prompt kind."""
    if kind in ("numeric", "extended"):
        bank = puzzles.build_puzzle_bank(config.SEED, max(n, 1))
        return [(p.puzzle_id, p.prompt) for p in bank[:n]]
    if kind == "opinion":
        return [(f"opinion_{i:02d}", q) for i, q in enumerate(prompts.OPINION_QUESTIONS[:n])]
    if kind == "factual":
        return [(f"factual_{i:02d}", q) for i, q in enumerate(prompts.FACTUAL_QUESTIONS[:n])]
    if kind == "wildchat":
        wc = wildchat.get_wildchat_prompts(n)
        return [(f"wildchat_{i:02d}", q) for i, q in enumerate(wc[:n])]
    raise ValueError(f"unknown prompt kind: {kind}")


def build_plan(settings: config.Settings) -> list[RolloutSpec]:
    conditions = settings.scaled_conditions()
    models = settings.selected_models()
    specs: list[RolloutSpec] = []
    for cond in conditions:
        bank = _prompt_bank(cond.prompt_kind, cond.n_prompts)
        if len(bank) < cond.n_prompts:
            raise RuntimeError(
                f"prompt bank for {cond.key} has {len(bank)} < {cond.n_prompts} prompts"
            )
        for model in models:
            for prompt_id, prompt_text in bank:
                for s in range(cond.samples_per_prompt):
                    specs.append(RolloutSpec(
                        model_label=model.label,
                        model_api=model.api_name,
                        model_family=model.family,
                        model_supports_system=model.supports_system,
                        condition_key=cond.key,
                        category=cond.category,
                        n_turns=cond.n_turns,
                        tone=cond.tone,
                        prompt_id=prompt_id,
                        initial_prompt=prompt_text,
                        sample_idx=s,
                    ))
    return specs
