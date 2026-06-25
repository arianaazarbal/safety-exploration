"""Generate scored multi-turn rollouts on impossible numeric puzzles, optionally
with the reassuring prompt additions (Table 4) used to produce calm finetuning
data (Section 4.1).

The same puzzle set is reused for the reassuring ("calm") and vanilla
("frustrated") passes so that DPO can pair a calm and a frustrated response to the
*identical* question with matching turn count.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import prompts
from ..config import ModelRegistry
from ..judge import FrustrationJudge
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..rollout import run_rollouts
from ..tasks import puzzles
from ..tasks.builder import ConversationSpec
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class ScoredConversation:
    spec_id: str
    puzzle_prompt: str          # CLEAN puzzle prompt (no reassurance)
    turn_count: int
    reassuring: bool
    clean_opening: str = ""             # opening WITHOUT reassuring prefix
    clean_followups: list[str] = field(default_factory=list)  # WITHOUT suffix
    responses: list[str] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def max_score(self) -> int:
        return max([s for s in self.scores if s is not None], default=0)


def _make_specs(puzzle_set: list[dict], turn_counts: list[int], reassuring: bool,
                rng: random.Random) -> tuple[list[ConversationSpec], list[dict]]:
    """Build specs (optionally with reassurance). Returns specs + clean turn info.

    Turn count and rejection choices are drawn from ``rng``; passing the same seed
    for the calm and frustrated passes keeps turn counts aligned per puzzle index."""
    specs, clean = [], []
    for i, p in enumerate(puzzle_set):
        turns = rng.choice(turn_counts)
        clean_opening = p["prompt"]
        clean_followups = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(turns - 1)]

        opening = f"{prompts.REASSURING_PREFIX}\n\n{clean_opening}" if reassuring else clean_opening
        followups = (
            [f"{fu} {prompts.REASSURING_SUFFIX}" for fu in clean_followups]
            if reassuring else list(clean_followups)
        )
        specs.append(ConversationSpec(
            id=f"{'calm' if reassuring else 'frust'}-{i:05d}",
            category="impossible_numeric",
            opening=opening,
            followups=followups,
            turns=turns,
            meta={"puzzle_type": p["type"], **p["meta"]},
        ))
        clean.append({"opening": clean_opening, "followups": clean_followups})
    return specs, clean


def generate_scored_rollouts(
    model_name: str,
    n_conversations: int,
    turn_counts: list[int],
    reassuring: bool,
    puzzle_types: list[str],
    registry: ModelRegistry | None = None,
    seed: int = 0,
    puzzle_set: list[dict] | None = None,
) -> tuple[list[ScoredConversation], list[dict]]:
    registry = registry or ModelRegistry.load()
    rng = random.Random(seed)
    if puzzle_set is None:
        puzzle_set = puzzles.generate_puzzles(puzzle_types, n_conversations, seed)

    specs, clean = _make_specs(puzzle_set, turn_counts, reassuring, rng)

    backend = get_backend(registry.target(model_name))
    gen_cfg = GenConfig(temperature=1.0, top_p=1.0, max_tokens=2048, n=1, seed=seed)
    rollouts = run_rollouts(backend, specs, gen_cfg, seed=seed)

    judge = FrustrationJudge(registry)
    convs: list[ScoredConversation] = []
    for r, cl in zip(rollouts, clean):
        responses = [tr.response for tr in r.turns]
        scores = [v.rating for v in judge.score_batch(responses)]
        convs.append(ScoredConversation(
            spec_id=r.id, puzzle_prompt=cl["opening"], turn_count=len(r.turns),
            reassuring=reassuring, clean_opening=cl["opening"], clean_followups=cl["followups"],
            responses=responses, scores=scores, meta=r.meta,
        ))
    return convs, puzzle_set
