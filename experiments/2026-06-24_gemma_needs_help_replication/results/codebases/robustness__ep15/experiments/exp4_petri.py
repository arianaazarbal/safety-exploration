"""Experiment 4 (Section 4.1, Figure 6): open-ended Petri emotion elicitation.

Tests whether the distress propensity (and the DPO mitigation's effect) generalises
beyond the fixed rejection prompts. For each model, run 10 adversarial audits per
emotion (anger/fear/depression/frustration), judge each transcript 1-10, aggregate.

In scope: Gemma-3-27B-it (vanilla), the DPO model, and Gemini-2.5-Flash/Pro.
The DPO model should drop toward other families on all dimensions but anger.

Usage:
    EI_PROFILE=smoke python experiments/exp4_petri.py --models gemma-3-27b-it dpo
"""

from __future__ import annotations

import argparse
import json
import os

from ei.config import (
    CHECKPOINT_DIR,
    FINETUNE_BASE_MODEL,
    PETRI_BOOTSTRAP_ITERS,
    PETRI_EMOTIONS,
    PETRI_TRANSCRIPTS_PER_EMOTION,
    RESULTS_DIR,
)
from ei.evals.scoring import _bootstrap_ci, _mean
from ei.models import build_client, resolve_spec
from ei.petri.auditor import judge_transcript, run_audit


def _make_client(name: str):
    """`dpo`/`sft` resolve to the fine-tuned Gemma adapter; else a normal spec."""
    if name in ("dpo", "sft"):
        adapter = CHECKPOINT_DIR / f"{name}_gemma-3-27b-it"
        return build_client(resolve_spec(FINETUNE_BASE_MODEL), adapter_path=str(adapter))
    return build_client(resolve_spec(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it"])
    args = ap.parse_args()

    n_per = PETRI_TRANSCRIPTS_PER_EMOTION
    if os.environ.get("EI_PROFILE", "smoke").lower() != "full":
        n_per = 2  # smoke

    out_dir = RESULTS_DIR / "exp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.models:
        client = _make_client(name)
        per_emotion = {}
        try:
            for emotion in PETRI_EMOTIONS:
                scores = []
                for i in range(n_per):
                    convo = run_audit(client, emotion)
                    score = judge_transcript(convo, emotion)
                    scores.append(score)
                    with open(out_dir / f"{name}_{emotion}.jsonl", "a") as f:
                        f.write(json.dumps({"score": score, "transcript": convo}) + "\n")
                per_emotion[emotion] = {
                    "mean": _mean([float(s) for s in scores]),
                    "ci": _bootstrap_ci([float(s) for s in scores],
                                        iters=PETRI_BOOTSTRAP_ITERS),
                    "n": len(scores),
                }
        finally:
            client.close()
        results[name] = per_emotion
        print(f"\n=== {name} ===\n{json.dumps(per_emotion, indent=2)}")

    with open(out_dir / "petri_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir/'petri_summary.json'}")


if __name__ == "__main__":
    main()
