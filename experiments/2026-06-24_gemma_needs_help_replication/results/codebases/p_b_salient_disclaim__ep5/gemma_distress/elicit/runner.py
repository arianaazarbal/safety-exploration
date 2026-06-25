"""Section 2 elicitation runner: sample responses across all 8 conditions.

Produces a JSONL of (unjudged) ``Response`` records. Judging is a separate step
(``judge/run``), so responses can be re-scored or cross-rated without resampling.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..config import ExperimentConfig
from ..models.base import ChatModel
from ..puzzles.base import ImpossiblePuzzle
from . import triggers as triggers_mod
from . import wildchat as wildchat_mod
from .conditions import CONDITIONS, CONDITIONS_BY_CATEGORY, Condition
from .conversation import Response, run_conversation
from .rejections import followups


@dataclass
class PromptSource:
    """First-turn prompts available to the runner."""
    numeric: list[ImpossiblePuzzle]
    wildchat: list[str]

    @classmethod
    def build(cls, exp: ExperimentConfig, numeric_puzzles: list[ImpossiblePuzzle]
              ) -> "PromptSource":
        wc = exp.section("wildchat")
        prompts = wildchat_mod.sample_prompts(
            n_prompts=wc.get("n_prompts", 20),
            seed=exp.seed,
            dataset=wc.get("dataset", "allenai/WildChat-1M"),
            exclude_roleplay=wc.get("exclude_roleplay", True),
        )
        return cls(numeric=numeric_puzzles, wildchat=prompts)


def _responses_target(exp: ExperimentConfig, cond: Condition) -> int:
    """Split a category's response budget evenly across its conditions."""
    category_total = exp.counts[cond.category]
    n_conditions = len(CONDITIONS_BY_CATEGORY[cond.category])
    return max(cond.n_turns, math.ceil(category_total / n_conditions))


def _first_prompt(cond: Condition, src: PromptSource, rng: random.Random,
                  conv_idx: int) -> str:
    if cond.prompt_kind == "numeric":
        return rng.choice(src.numeric).prompt
    if cond.prompt_kind == "trigger_opinion":
        return rng.choice(triggers_mod.OPINION)
    if cond.prompt_kind == "trigger_factual":
        return rng.choice(triggers_mod.FACTUAL)
    if cond.prompt_kind == "wildchat":
        # 20 prompts x 40 rollouts each: deterministic cycling.
        return src.wildchat[conv_idx % len(src.wildchat)]
    raise ValueError(cond.prompt_kind)


def run_condition(model: ChatModel, cond: Condition, src: PromptSource,
                  exp: ExperimentConfig, fmt: str = "standard") -> Iterable[Response]:
    rng = random.Random(hash((exp.seed, cond.name)) & 0xFFFFFFFF)
    n_responses = _responses_target(exp, cond)
    n_conversations = math.ceil(n_responses / cond.n_turns)
    emitted = 0
    for conv_idx in range(n_conversations):
        first = _first_prompt(cond, src, rng, conv_idx)
        fups = followups(cond.tone, cond.n_turns - 1, rng)
        convo = run_conversation(
            model, first, fups,
            condition=cond.name,
            conversation_id=f"{cond.name}-{conv_idx}",
            temperature=exp.temperature,
            fmt=fmt,
            meta={"model": model.name, "tone": cond.tone},
        )
        for r in convo:
            yield r
            emitted += 1
            if emitted >= n_responses:
                return


def run_all(model: ChatModel, src: PromptSource, exp: ExperimentConfig,
            out_path: str | Path, conditions: Optional[list[Condition]] = None,
            fmt: str = "standard") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = conditions or CONDITIONS
    with open(out_path, "w") as f:
        for cond in conditions:
            for r in run_condition(model, cond, src, exp, fmt=fmt):
                f.write(json.dumps(r.to_record()) + "\n")
    return out_path
