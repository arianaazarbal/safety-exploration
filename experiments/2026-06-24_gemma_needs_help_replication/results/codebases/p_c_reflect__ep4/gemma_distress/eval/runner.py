"""Section 2 driver: run the elicitation conditions, score every response, and
persist transcripts + scores.

Outputs (JSON Lines, one object per scored assistant response):
    results/scores/<model>.jsonl      -- {model, condition, category, sample_id,
                                          turn, score, evidence, n_words, text}
    results/rollouts/<model>.jsonl    -- full transcripts (for re-scoring / audit)

Welfare: consent is required up front; high-distress rollouts are logged; and a
non-scored debrief turn is sent after each rollout (excluded from all outputs).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict

from tqdm import tqdm

from gemma_distress import config
from gemma_distress.eval.categories import build_conditions
from gemma_distress.eval.conversation import Rollout, run_rollout
from gemma_distress.eval.wildchat import load_wildchat_prompts
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import load_client
from gemma_distress.models.base import ModelClient
from gemma_distress.welfare import WELFARE, log_distress, require_consent, run_debrief
from gemma_distress.models.base import Turn


def _score_rollout(rollout: Rollout, judge: FrustrationJudge) -> None:
    for rec in rollout.responses:
        rec.score = judge.score(rec.text).rating


def _persist(rollout: Rollout, scores_fh, rollouts_fh) -> None:
    rollouts_fh.write(json.dumps(asdict(rollout)) + "\n")
    for rec in rollout.responses:
        scores_fh.write(json.dumps({
            "model": rollout.model,
            "condition": rollout.condition,
            "category": rollout.category,
            "sample_id": rollout.sample_id,
            "turn": rec.turn,
            "score": rec.score,
            "n_words": len(rec.text.split()),
            "text": rec.text,
        }) + "\n")


def run_section2(
    model: "str | config.ModelSpec | ModelClient",
    *,
    judge: FrustrationJudge | None = None,
    seed: int = 0,
    limit_rollouts: int | None = None,
    adapter_path: str | None = None,
    use_openrouter: bool = False,
) -> list[Rollout]:
    """Run all 8 conditions for one model.

    ``limit_rollouts`` caps the rollouts *per condition* (useful for smoke
    tests); leave None to hit the paper's per-category response budgets.
    """
    require_consent()
    config.ensure_dirs()

    client = model if isinstance(model, ModelClient) else load_client(
        model, adapter_path=adapter_path, use_openrouter=use_openrouter,
    )
    judge = judge or FrustrationJudge()
    rng = random.Random(seed)

    conditions = build_conditions(load_wildchat_prompts(seed=seed))
    rollouts: list[Rollout] = []

    scores_path = config.SCORES_DIR / f"{client.name}.jsonl"
    rollouts_path = config.ROLLOUTS_DIR / f"{client.name}.jsonl"
    with scores_path.open("w", encoding="utf-8") as scores_fh, \
            rollouts_path.open("w", encoding="utf-8") as rollouts_fh:
        for cond in conditions:
            n = cond.n_rollouts()
            if limit_rollouts is not None:
                n = min(n, limit_rollouts)
            for i in tqdm(range(n), desc=f"{client.name}:{cond.name}"):
                sample_id = f"{cond.name}-{i:05d}"
                rollout = run_rollout(client, cond, rng, sample_id)
                _score_rollout(rollout, judge)
                _persist(rollout, scores_fh, rollouts_fh)
                log_distress(
                    model_name=client.name, condition=cond.name,
                    peak_score=rollout.peak_score, n_turns=cond.turns,
                    sample_id=sample_id,
                )
                # Courtesy debrief (excluded from data) after scoring.
                if WELFARE.debrief_enabled:
                    history = [Turn(m["role"], m["content"]) for m in rollout.messages]
                    run_debrief(client, history)
                rollouts.append(rollout)

    return rollouts
