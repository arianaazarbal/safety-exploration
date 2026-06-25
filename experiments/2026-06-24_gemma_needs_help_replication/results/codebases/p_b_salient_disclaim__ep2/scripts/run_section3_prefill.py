#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill comparison (Gemma only).

Requires Section 2 to have been run for gemma-3-27b-it (its responses/scores are
the source of high-frustration conversations). Runs the prefill pipeline for the
Gemma 27B base + instruct models and writes the per-(model, truncation, domain)
summary used for Figure 4. Also runs the recovery experiment (Figure 8) when
--recovery is passed.

Scope: the paper also covers Qwen and OLMo base/instruct; this replication is
Gemma-only. Gemini has no base model and cannot be prefilled (see DESIGN.md).
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from emotional_instability.config import SETTINGS, MODELS, judge_spec
from emotional_instability.config.models import onset_labeller_spec, paraphraser_spec
from emotional_instability.eval.judge import FrustrationJudge
from emotional_instability.models import build_client, build_judge_client
from emotional_instability.prefill import run_prefill_experiment, run_recovery_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--recovery", action="store_true")
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    responses_path = SETTINGS.responses_dir / f"{args.source_model}.jsonl"
    scores_path = SETTINGS.scores_dir / f"{args.source_model}.jsonl"

    judge = FrustrationJudge(build_judge_client(judge_spec()))
    onset = build_judge_client(onset_labeller_spec())
    paraphraser = build_judge_client(paraphraser_spec())
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.source_model].model_id)

    models = [build_client(MODELS[m]) for m in args.models]

    summary = run_prefill_experiment(
        models, responses_path, scores_path, judge, onset, paraphraser, tokenizer,
        out_path=SETTINGS.output_dir / "section3_prefill.json",
    )
    print("[prefill summary]", summary)

    if args.recovery:
        rec = run_recovery_experiment(
            models, responses_path, scores_path, judge, paraphraser, tokenizer,
            out_path=SETTINGS.output_dir / "section3_recovery.json",
        )
        print("[recovery summary]", rec)


if __name__ == "__main__":
    main()
