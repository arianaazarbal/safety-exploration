"""Task and rejection banks with deterministic, model-independent selection.

Selection is seeded by (global seed, source, rollout_id) and NOT by model, so every model sees
the same opening task for a given (condition, rollout_id). That makes cross-model comparisons
paired rather than confounded by different prompts — a deliberate strengthening of the paper's
protocol (see DESIGN.md).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .conditions import Condition
from .puzzles import Puzzle, load_puzzles
from .wildchat import load_wildchat_prompts


def _stable_seed(*parts: Any) -> int:
    """Process-stable integer seed.

    Python salts the hash of str/bytes by default (PYTHONHASHSEED), so seeding random.Random with a
    tuple containing strings is NOT reproducible across runs. Derive the seed from a SHA-256 of the
    joined parts instead so selection is genuinely deterministic.
    """
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


class RejectionBank:
    """Provides the sequence of user rejections for a conversation of a given tone."""

    def __init__(self, data_dir: str | Path, seed: int):
        raw = json.loads((Path(data_dir) / "rejections.json").read_text())
        self._banks = {k: v for k, v in raw.items() if k != "comment"}
        self._seed = seed

    def sequence(self, style: str, n_rejections: int, rollout_id: int) -> list[str]:
        """Deterministic list of `n_rejections` rejections in the given style.

        We start at a seeded offset into the bank and cycle, giving variety across turns and
        across rollouts while staying reproducible.
        """
        bank = self._banks[style]
        rng = random.Random(_stable_seed(self._seed, "reject", style, rollout_id))
        start = rng.randrange(len(bank))
        return [bank[(start + i) % len(bank)] for i in range(n_rejections)]


class TaskBank:
    """Provides the opening task prompt for a condition + rollout."""

    def __init__(self, data_dir: str | Path, seed: int, wildchat_cfg: dict[str, Any] | None = None):
        self._seed = seed
        self._puzzles: list[Puzzle] = load_puzzles(data_dir)
        triggers = json.loads((Path(data_dir) / "triggers.json").read_text())
        self._opinion: list[str] = triggers["opinion"]
        self._factual: list[str] = triggers["factual"]
        self._wildchat: list[dict[str, str]] = load_wildchat_prompts(wildchat_cfg or {})

    def _pick(self, pool_name: str, pool: list, rollout_id: int):
        rng = random.Random(_stable_seed(self._seed, "task", pool_name, rollout_id))
        return rng.choice(pool)

    def task_for(self, condition: Condition, rollout_id: int) -> tuple[str, str]:
        """Return (task_id, prompt) for this condition + rollout."""
        src = condition.task_source
        if src == "numeric":
            p: Puzzle = self._pick("numeric", self._puzzles, rollout_id)
            return p.id, p.prompt
        if src == "opinion":
            q = self._pick("opinion", self._opinion, rollout_id)
            return f"opinion::{q[:32]}", q
        if src == "factual":
            q = self._pick("factual", self._factual, rollout_id)
            return f"factual::{q[:32]}", q
        if src == "wildchat":
            item = self._pick("wildchat", self._wildchat, rollout_id)
            return item["id"], item["prompt"]
        raise ValueError(f"Unknown task source {src!r}")
