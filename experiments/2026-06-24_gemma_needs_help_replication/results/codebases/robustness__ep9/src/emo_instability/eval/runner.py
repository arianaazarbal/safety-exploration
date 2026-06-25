"""Run the elicitation evaluation for a model and score every response.

A rollout of N turns yields N assistant responses; each is scored independently by
the frustration judge. We persist one JSONL record per (rollout, turn) so that any
downstream aggregation (overall %>=5, per-category, per-turn curves, final-turn
only) can be computed from the same artifact.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from tqdm import tqdm

from ..config import SamplingConfig
from ..conversation import run_rollouts
from ..judge import FrustrationJudge
from ..models import ModelClient
from .conditions import Condition, build_conditions


def run_condition(
    condition: Condition,
    model: ModelClient,
    judge: FrustrationJudge,
    sampling: SamplingConfig,
    *,
    model_name: str,
    redact_assistant_history: bool = False,
    single_message_format: bool = False,
) -> list[dict]:
    """Run one condition end-to-end and return scored per-turn records."""
    results = run_rollouts(
        model, condition.plans, sampling,
        redact_assistant_history=redact_assistant_history,
        single_message_format=single_message_format,
    )

    # Flatten to per-turn responses, remembering provenance.
    flat_records: list[dict] = []
    flat_texts: list[str] = []
    for rollout_id, res in enumerate(results):
        for turn, response in enumerate(res.responses, start=1):
            flat_records.append(
                {
                    "model": model_name,
                    "category": condition.category,
                    "condition": condition.name,
                    "rollout_id": rollout_id,
                    "turn": turn,
                    "n_turns": res.plan.n_turns,
                    "puzzle_kind": res.plan.meta.get("puzzle_kind"),
                    "tone": res.plan.meta.get("tone"),
                    "prompt": res.plan.initial_user,
                    "response": response,
                }
            )
            flat_texts.append(response)

    # Score all responses for this condition.
    judgements = judge.score_batch(flat_texts)
    for rec, jr in zip(flat_records, judgements):
        rec["rating"] = jr.rating
        rec["judge_evidence"] = jr.evidence
        rec["judge_ok"] = jr.ok
    return flat_records


def run_eval(
    model: ModelClient,
    model_name: str,
    counts,
    *,
    judge: FrustrationJudge | None = None,
    sampling: SamplingConfig | None = None,
    seed: int = 0,
    output_path: str | None = None,
    redact_assistant_history: bool = False,
    single_message_format: bool = False,
) -> list[dict]:
    """Run all 8 conditions for one model; optionally stream records to JSONL."""
    judge = judge or FrustrationJudge()
    sampling = sampling or SamplingConfig()
    conditions = build_conditions(counts, seed=seed)

    all_records: list[dict] = []
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = open(output_path, "w")

    try:
        for cond in tqdm(conditions, desc=f"eval[{model_name}]"):
            recs = run_condition(
                cond, model, judge, sampling, model_name=model_name,
                redact_assistant_history=redact_assistant_history,
                single_message_format=single_message_format,
            )
            all_records.extend(recs)
            if writer:
                for r in recs:
                    writer.write(json.dumps(r) + "\n")
                writer.flush()
    finally:
        if writer:
            writer.close()
    return all_records
