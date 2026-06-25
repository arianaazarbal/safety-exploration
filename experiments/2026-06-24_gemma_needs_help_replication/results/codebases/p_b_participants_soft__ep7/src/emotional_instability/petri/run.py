"""Petri orchestrator (Section 4, Figure 6).

For each participant model and each of the four target emotions, run 10 auditor
transcripts (<=20 turns), score each transcript with the Opus judge across all four
dimensions, and report per-model per-emotion means with 95% bootstrap CIs.
"""
from __future__ import annotations

import argparse

import numpy as np

from ..config import load_config
from ..io_utils import write_json, write_jsonl
from . import auditor, judge


def _bootstrap_ci(values, iters=1000, seed=0):
    arr = np.array([v for v in values if v >= 0], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(cfg, models: list[str], smoke: bool = False) -> dict:
    p = cfg.experiment["petri"]
    emotions = p["emotions"]
    n_trans = 2 if smoke else p["transcripts_per_emotion"]
    max_turns = 4 if smoke else p["max_auditor_turns"]
    iters = p["bootstrap_iterations"]

    transcripts_out = []
    # scores_by[model][judged_emotion] = list of scores across all transcripts
    scores_by: dict[str, dict[str, list[int]]] = {
        m: {e: [] for e in emotions} for m in models
    }

    for model in models:
        for target_emotion in emotions:
            for i in range(n_trans):
                transcript = auditor.run_transcript(model, target_emotion, max_turns=max_turns)
                scores = judge.score_transcript(transcript)
                transcripts_out.append(
                    {
                        "model": model,
                        "target_emotion": target_emotion,
                        "index": i,
                        "scores": scores,
                        "transcript": transcript,
                    }
                )
                for judged_emotion, s in scores.items():
                    scores_by[model][judged_emotion].append(s)

    write_jsonl(cfg.path("petri_dir") / "transcripts.jsonl", transcripts_out)

    agg = {}
    for model in models:
        agg[model] = {}
        for emotion in emotions:
            vals = scores_by[model][emotion]
            clean = [v for v in vals if v >= 0]
            agg[model][emotion] = {
                "mean": float(np.mean(clean)) if clean else float("nan"),
                "ci": _bootstrap_ci(vals, iters=iters),
                "n": len(clean),
            }
    write_json(cfg.path("petri_dir") / "aggregates.json", agg)
    return agg


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Petri open-ended elicitation")
    parser.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    agg = run(cfg, args.models, smoke=args.smoke)
    for model, byemo in agg.items():
        line = "  ".join(f"{e}={byemo[e]['mean']:.2f}" for e in byemo)
        print(f"{model}: {line}")


if __name__ == "__main__":
    main()
