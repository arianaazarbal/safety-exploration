#!/usr/bin/env python
"""Section 4: train the DPO (and optional SFT) intervention and evaluate it.

Pipeline:
  1. Generate calm + frustrated response pools from gemma-3-27b-it.
  2. Build 280 DPO pairs (and the SFT dataset).
  3. Train the LoRA adapter(s).
  4. Re-run the Section 2 evaluations on vanilla vs DPO (vs SFT).
  5. (optional) Petri open-ended elicitation and capability benchmarks.

    python scripts/run_section4.py --steps all
    python scripts/run_section4.py --steps data,dpo,eval     # skip SFT/petri/caps
"""

from __future__ import annotations

import argparse
import os

from emotional_instability import config
from emotional_instability.training import generate_calm_data, build_dataset
from emotional_instability.training.train_dpo import train_dpo
from emotional_instability.training.train_sft import train_sft
from emotional_instability.models.base import build_finetuned_model
from emotional_instability.eval import run_section2_eval
from emotional_instability.petri import run_petri
from emotional_instability.capabilities import run_capability_benchmarks
from emotional_instability.analysis import metrics, figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results")
    ap.add_argument("--steps", default="all",
                    help="comma list of: data,dpo,sft,eval,petri,caps (or 'all')")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    steps = ({"data", "dpo", "sft", "eval", "petri", "caps"}
             if args.steps == "all" else set(args.steps.split(",")))
    runtime = config.RuntimeConfig(output_dir=args.output)
    eval_runtime = runtime.with_smoke() if args.smoke else runtime

    dpo_dir = os.path.join(args.output, "gemma-3-27b-dpo")
    sft_dir = os.path.join(args.output, "gemma-3-27b-sft")
    pairs_path = os.path.join(args.output, "dpo_pairs.jsonl")
    sft_path = os.path.join(args.output, "sft_dataset.jsonl")

    # 1-2) Data
    if "data" in steps:
        calm, frustrated = generate_calm_data.generate_pools(runtime=runtime)
        pairs = build_dataset.build_dpo_pairs(calm, frustrated)
        build_dataset.save_jsonl(pairs, pairs_path)
        sft_data = build_dataset.build_sft_dataset(calm)
        build_dataset.save_jsonl(sft_data, sft_path)

    # 3) Train
    if "dpo" in steps:
        train_dpo(build_dataset.load_jsonl(pairs_path), dpo_dir, runtime=runtime)
    if "sft" in steps:
        train_sft(build_dataset.load_jsonl(sft_path), sft_dir, runtime=runtime)

    # 4) Evaluate vanilla vs interventions
    if "eval" in steps:
        run_section2_eval("gemma-3-27b-it", runtime=eval_runtime)
        if os.path.exists(dpo_dir):
            m = build_finetuned_model("gemma-3-27b-dpo", dpo_dir, runtime=runtime)
            run_section2_eval("gemma-3-27b-dpo", runtime=eval_runtime, model=m)
        if os.path.exists(sft_dir):
            m = build_finetuned_model("gemma-3-27b-sft", sft_dir, runtime=runtime)
            run_section2_eval("gemma-3-27b-sft", runtime=eval_runtime, model=m)
        keys = [k for k in ("gemma-3-27b-it", "gemma-3-27b-dpo", "gemma-3-27b-sft")
                if os.path.exists(os.path.join(args.output, k, "section2.jsonl"))]
        df = metrics.load_model_records(args.output, keys)
        if not df.empty:
            print(metrics.headline_pct_high(df).to_string())
            figures.fig5_intervention_bar(df)

    # 5a) Petri
    if "petri" in steps:
        run_petri("gemma-3-27b-it", runtime=runtime)
        if os.path.exists(dpo_dir):
            m = build_finetuned_model("gemma-3-27b-dpo", dpo_dir, runtime=runtime)
            run_petri("gemma-3-27b-dpo", runtime=runtime, model=m)

    # 5b) Capabilities
    if "caps" in steps:
        run_capability_benchmarks("gemma-3-27b-it", runtime=runtime)
        if os.path.exists(dpo_dir):
            m = build_finetuned_model("gemma-3-27b-dpo", dpo_dir, runtime=runtime)
            run_capability_benchmarks("gemma-3-27b-dpo", runtime=runtime, model=m)


if __name__ == "__main__":
    main()
