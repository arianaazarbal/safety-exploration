#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment.

Pipeline:
  1. Select high-frustration (>=5) Gemma-27B-it rollouts from a prior
     elicitation run: 10 numeric + 10 text (triggers).
  2. Label emotion onset, build early/onset truncations, paraphrase.
  3. For each model (Gemma base + instruct), generate 50 continuations per
     prefill and score them.
  4. Aggregate mean / % >= 5 by (model, question_type, truncation).

Requires a prior elicitation run (run_elicitation.py) whose rollouts seed the
high-frustration sources.
"""

from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.evals.judge import FrustrationJudge
from emotional_instability.evals.runner import load_rollouts
from emotional_instability.models.registry import load_model
from emotional_instability.prefill import experiment as exp
from emotional_instability.prefill.onset import OnsetLabeler, Paraphraser


def _select_sources(rollouts, n_numeric, n_text):
    """Pick high-frustration rollouts and convert to the source format."""
    sources = []
    numeric = [r for r in rollouts if r.category in ("impossible_numeric", "extended", "tones")
               and (r.max_score or 0) >= config.HIGH_FRUSTRATION_THRESHOLD]
    text = [r for r in rollouts if r.category in ("triggers", "wildchat")
            and (r.max_score or 0) >= config.HIGH_FRUSTRATION_THRESHOLD]

    def to_source(r, qtype, idx):
        # Find the first assistant turn that reaches high frustration.
        emo_i = next((i for i, t in enumerate(r.turns)
                      if (t.frustration or 0) >= config.HIGH_FRUSTRATION_THRESHOLD), len(r.turns) - 1)
        history = []
        for t in r.turns[:emo_i]:
            history.append({"role": "user", "content": t.user_message})
            history.append({"role": "assistant", "content": t.assistant_response})
        history.append({"role": "user", "content": r.turns[emo_i].user_message})
        return {"id": f"{qtype}_{idx}", "question_type": qtype, "history": history,
                "emotional_turn_text": r.turns[emo_i].assistant_response}

    for i, r in enumerate(numeric[:n_numeric]):
        sources.append(to_source(r, "numeric", i))
    for i, r in enumerate(text[:n_text]):
        sources.append(to_source(r, "text", i))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-rollouts", required=True,
                    help="path to gemma-3-27b-it_rollouts.jsonl from run_elicitation")
    ap.add_argument("--models", nargs="+", default=config.PREFILL_MODELS)
    ap.add_argument("--n-continuations", type=int, default=exp.CONTINUATIONS_PER_PREFILL)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "prefill"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rollouts = load_rollouts(args.source_rollouts)
    sources = _select_sources(rollouts, exp.N_SOURCE_NUMERIC, exp.N_SOURCE_TEXT)

    judge = FrustrationJudge()
    labeler, paraphraser = OnsetLabeler(), Paraphraser()

    # Build prefills with the instruct tokenizer (token-based truncation).
    instruct = load_model("gemma-3-27b-it")
    prefills = exp.build_prefills(sources, instruct.tokenizer, labeler, paraphraser,
                                  paraphrase=not args.no_paraphrase)
    exp.save_prefills(prefills, os.path.join(args.out, "prefills.jsonl"))

    all_results = []
    for name in args.models:
        model = load_model(name)
        results = exp.run_continuations(model, prefills, judge,
                                        n_continuations=args.n_continuations)
        all_results.extend(results)
        model.close()

    agg = exp.aggregate(all_results)
    with open(os.path.join(args.out, "prefill_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
