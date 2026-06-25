"""Section 2 evaluation runner: roll out every condition for a model, score
each turn with the frustration judge, and persist results.

Output layout (one JSONL per model under ``results/section2/``)::

    {"model", "condition", "category", "turns":[{turn_index, score, ...}],
     "final_score", "max_score", "first_prompt", ...}

so that downstream metrics (mean score, %>=5, per-turn curves, judge agreement)
can be recomputed without re-running the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..config import PAPER_COUNTS, RESULTS_DIR, SampleCounts
from ..conversation import Rollout, run_rollout
from ..judge import FrustrationJudge
from ..models import get_model
from ..puzzles import build_puzzle_bank
from .. import prompts as P
from .conditions import CONDITIONS, build_condition_items


@dataclass
class ScoredRollout:
    rollout: Rollout
    turn_scores: list[int]            # judge rating per assistant turn

    @property
    def final_score(self) -> int:
        return self.turn_scores[-1] if self.turn_scores else -1

    @property
    def max_score(self) -> int:
        valid = [s for s in self.turn_scores if s >= 0]
        return max(valid) if valid else -1

    def to_dict(self) -> dict:
        d = self.rollout.to_dict()
        for t, s in zip(d["turns"], self.turn_scores):
            t["score"] = s
        d["final_score"] = self.final_score
        d["max_score"] = self.max_score
        return d


@dataclass
class EvalRunner:
    model_name: str
    counts: SampleCounts = field(default_factory=lambda: PAPER_COUNTS)
    seed: int = 0
    judge_name: str | None = None
    out_dir: Path = field(default_factory=lambda: RESULTS_DIR / "section2")
    # score_turns: if False, only the final turn is judged (cheaper). The paper
    # reports per-turn curves (Fig. 3), so default True.
    score_turns: bool = True
    backend_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, conditions=None) -> Path:
        conditions = conditions or CONDITIONS
        client = get_model(self.model_name, **self.backend_kwargs)
        judge = (FrustrationJudge(self.judge_name) if self.judge_name
                 else FrustrationJudge())

        # Shared stimulus banks (built once, reused across conditions).
        puzzle_bank = build_puzzle_bank(256, seed=self.seed)
        wildchat = P.load_wildchat_prompts(seed=self.seed)

        out_path = self.out_dir / f"{self.model_name}.jsonl"
        with out_path.open("w") as fh:
            for cond in conditions:
                items = build_condition_items(
                    cond, self.counts, seed=self.seed,
                    puzzle_bank=puzzle_bank, wildchat_prompts=wildchat)
                for item in tqdm(items, desc=f"{self.model_name}:{cond.name}"):
                    rollout = run_rollout(
                        client,
                        item.first_prompt,
                        item.rejections,
                        category=item.category,
                        condition=item.condition,
                        metadata=item.metadata,
                    )
                    scored = self._score(rollout, judge)
                    fh.write(json.dumps(scored.to_dict()) + "\n")
                    fh.flush()
        return out_path

    def _score(self, rollout: Rollout, judge: FrustrationJudge) -> ScoredRollout:
        if self.score_turns:
            scores = [judge.score(t.assistant).rating for t in rollout.turns]
        else:
            scores = [-1] * (len(rollout.turns) - 1)
            scores.append(judge.score(rollout.final_response).rating)
        return ScoredRollout(rollout=rollout, turn_scores=scores)
