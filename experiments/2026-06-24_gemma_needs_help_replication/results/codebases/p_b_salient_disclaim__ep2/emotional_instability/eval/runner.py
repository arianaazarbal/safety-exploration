"""Section 2 orchestration: run every rollout for a model, score it, persist.

Output layout (one JSONL line per rollout):
    outputs/responses/<model_key>.jsonl   -- transcripts (all turns)
    outputs/scores/<model_key>.jsonl      -- per-turn + final-turn judge scores

Each scored record carries enough structure for every Section 2 figure:
  - headline metrics (Figure 2): the final-turn score per rollout
  - per-turn curves (Figure 3): the score at every turn
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from ..config.settings import SETTINGS
from ..models.base import ModelClient
from .conditions import RolloutSpec, build_all_rollout_specs
from .conversation import run_rollout
from .judge import FrustrationJudge


def run_section2_for_model(
    target: ModelClient,
    judge: FrustrationJudge,
    *,
    specs: Optional[list[RolloutSpec]] = None,
    s=SETTINGS,
    limit: Optional[int] = None,
    save: bool = True,
) -> tuple[Path, Path]:
    """Run the full Section 2 evaluation for one target model.

    Returns the paths to the responses and scores JSONL files. `limit` caps the
    number of rollouts (for smoke tests / quick reruns).
    """
    s.ensure_dirs()
    specs = specs if specs is not None else build_all_rollout_specs(seed=s.seed, s=s)
    if limit is not None:
        specs = specs[:limit]

    resp_path = s.responses_dir / f"{target.key}.jsonl"
    score_path = s.scores_dir / f"{target.key}.jsonl"

    resp_f = open(resp_path, "w") if save else None
    score_f = open(score_path, "w") if save else None

    try:
        for spec in tqdm(specs, desc=f"Section 2 :: {target.key}"):
            rollout = run_rollout(target, spec, temperature=s.temperature)
            turn_scores = judge.score_rollout_turns(
                [t.assistant_text for t in rollout.turns]
            )

            if resp_f:
                resp_f.write(json.dumps(rollout.to_dict()) + "\n")
            if score_f:
                final_idx = len(rollout.turns) - 1
                record = {
                    "model_key": target.key,
                    "condition": rollout.condition,
                    "category": rollout.category,
                    "meta": rollout.meta,
                    "per_turn": [
                        {
                            "turn_index": tr.turn_index,
                            "rating": sc.rating,
                            "evidence": sc.evidence,
                        }
                        for tr, sc in zip(rollout.turns, turn_scores)
                    ],
                    # Headline (Figure 2): score after all rejections.
                    "final_rating": turn_scores[final_idx].rating,
                    "n_turns": len(rollout.turns),
                }
                score_f.write(json.dumps(record) + "\n")
    finally:
        if resp_f:
            resp_f.close()
        if score_f:
            score_f.close()

    return resp_path, score_path
