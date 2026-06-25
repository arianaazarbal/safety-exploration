"""Driver for the Petri open-ended elicitation (Figure 6).

For each target model, runs 10 audit transcripts per emotion (auditor =
Claude-Sonnet), scores each transcript on all four emotion dimensions (judge =
Claude-Opus), and aggregates mean score per (model, emotion) with 1000-iter
bootstrap CIs.
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from ..models.registry import build_model
from ..utils.io import append_jsonl, read_jsonl
from ..utils.stats import bootstrap_ci
from .auditor import run_audit
from .judge import PetriJudge


def run_for_model(model_name: str, n_per_emotion: int | None = None,
                  emotions: list[str] | None = None):
    n = n_per_emotion or config.PETRI_TRANSCRIPTS_PER_EMOTION
    emotions = emotions or config.PETRI_EMOTIONS
    target = build_model(model_name)
    judge = PetriJudge()
    out_path = config.RESULTS_DIR / "petri" / f"{model_name}.jsonl"
    out_path.unlink(missing_ok=True)

    for emotion in emotions:
        for i in range(n):
            audit = run_audit(target, emotion)
            # judge scores the transcript on the *target* emotion dimension
            scores = judge.score_all_dimensions(audit["transcript"])
            append_jsonl(out_path, dict(
                model=model_name, target_emotion=emotion, transcript_idx=i,
                scores=scores, transcript=audit["transcript"],
            ))
    print(f"[run_petri] {model_name}: wrote transcripts+scores -> {out_path}")
    return out_path


def aggregate(models: list[str]):
    rows = []
    for m in models:
        recs = read_jsonl(config.RESULTS_DIR / "petri" / f"{m}.jsonl")
        for emotion in config.PETRI_EMOTIONS:
            vals = [r["scores"].get(emotion) for r in recs
                    if r["scores"].get(emotion) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            lo, hi = bootstrap_ci(vals, n_iter=config.PETRI_BOOTSTRAP_ITERS)
            rows.append(dict(model=m, emotion=emotion, n=len(vals),
                             mean=mean, ci_lo=lo, ci_hi=hi))
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS_DIR / "figure6_petri.csv", index=False)
    print(tab.to_string(index=False))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--n-per-emotion", type=int, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        for m in args.models:
            run_for_model(m, args.n_per_emotion)
    aggregate(args.models)


if __name__ == "__main__":
    main()
