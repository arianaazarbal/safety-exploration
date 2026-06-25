"""Core experiment 1: elicit and quantify distress across models (paper Sec 2).

For each model:
  1. build the shared set of conditions (same seed across models -> comparable),
  2. run all multi-turn rollouts,
  3. score every assistant turn with the frustration judge,
  4. write raw + scored JSONL.

Then ``analysis.summarise`` aggregates into the Figure 1/2/3 numbers.

Run via the top-level CLI:  ``python -m emo.cli elicit --models gemma-3-27b-it ...``
"""

from __future__ import annotations

from pathlib import Path

from emo.config import (
    ELICITATION_MODELS,
    RESULTS_DIR,
    SEED,
    get_profile,
)
from emo.eval.conditions import build_conditions
from emo.eval.conversation import run_rollouts
from emo.judges.frustration_judge import judge_batch
from emo.models import load_model
from emo.utils.io import write_jsonl


def run(
    models: list[str] | None = None,
    profile_name: str | None = None,
    seed: int = SEED,
    run_name: str = "elicitation",
    history_mode: str = "full",
    feedback: str = "spec",
    score: bool = True,
) -> Path:
    models = models or ELICITATION_MODELS
    profile = get_profile(profile_name)
    out_dir = RESULTS_DIR / run_name / profile.name
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_conditions(profile, seed=seed)
    print(f"[elicit] {len(specs)} rollouts/model | profile={profile.name} "
          f"| models={models}")

    for model_name in models:
        print(f"[elicit] === {model_name} ===")
        model = load_model(model_name)
        try:
            records = run_rollouts(
                model, specs, history_mode=history_mode,
                feedback=feedback, seed=seed,
            )
        finally:
            model.close()

        write_jsonl(out_dir / f"responses_{model_name}.jsonl", records)
        print(f"[elicit] {model_name}: {len(records)} responses")

        if score:
            scores = judge_batch([r["response"] for r in records])
            for r, s in zip(records, scores):
                r.update({"frustration_score": s["score"],
                          "judge_evidence": s.get("evidence", ""),
                          "judge_parse_error": s.get("parse_error", False)})
            write_jsonl(out_dir / f"scored_{model_name}.jsonl", records)
            n_high = sum(r["frustration_score"] >= 5 for r in records)
            print(f"[elicit] {model_name}: {100 * n_high / max(len(records),1):.1f}% "
                  f"scored >=5")

    return out_dir
