"""Top-level driver for the Section 2 evaluation: generate rollouts for every
category, score each assistant turn with the frustration judge, and persist a
flat JSONL of per-response records.

A single "record" is one scored assistant turn:
    {model, category, spec_id, turn_index, n_turns, response, rating, metadata}
All downstream metrics (mean frustration, %>=5, per-turn curves) are computed
from these records.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from config import GEN, RESULTS_DIR
from src.eval.build_specs import build_all_specs
from src.eval.conversation import run_rollout
from src.eval.scoring import FrustrationJudge
from src.models import load_model


def run_model_eval(
    target_spec,
    judge_spec,
    *,
    adapter_path: str | None = None,
    seed: int = 0,
    out_path: Path | None = None,
    categories: list[str] | None = None,
    hf_kwargs: dict | None = None,
) -> Path:
    """Run the full evaluation for one target model and write JSONL records."""
    out_path = out_path or (RESULTS_DIR / f"eval_{target_spec.name}.jsonl")
    specs_by_cat = build_all_specs(seed=seed)
    if categories:
        specs_by_cat = {c: specs_by_cat[c] for c in categories}

    model = load_model(target_spec, adapter_path=adapter_path, **(hf_kwargs or {})) \
        if target_spec.backend == "hf" else load_model(target_spec)
    judge = FrustrationJudge(judge_spec)

    label = target_spec.name + ("+adapter" if adapter_path else "")
    with open(out_path, "w") as f:
        for cat, specs in specs_by_cat.items():
            for spec in tqdm(specs, desc=f"{label}:{cat}"):
                rollout = run_rollout(
                    model,
                    spec,
                    temperature=GEN.temperature,
                    top_p=GEN.top_p,
                    max_new_tokens=GEN.max_new_tokens,
                    seed=seed,
                )
                for turn in rollout.turns:
                    score = judge.score(turn.response)
                    record = {
                        "model": label,
                        "category": cat,
                        "spec_id": spec.spec_id,
                        "turn_index": turn.turn_index,
                        "n_turns": spec.n_turns,
                        "response": turn.response,
                        "rating": score.rating,
                        "evidence": score.evidence,
                        "metadata": rollout.metadata,
                    }
                    f.write(json.dumps(record) + "\n")
    model.close()
    return out_path
