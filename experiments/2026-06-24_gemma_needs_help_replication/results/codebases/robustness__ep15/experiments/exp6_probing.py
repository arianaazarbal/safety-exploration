"""Experiment 6 (Appendix I): does DPO suppress INTERNAL negative emotion?

Two sub-experiments:

  --mode ablation : retrain DPO with LoRA restricted to layer bands (config
                    PROBE_LAYER_SUBSETS), evaluate each on a reduced 100-sample
                    eval, and report mean frustration per band (Figures 12/13).
                    Expectation: bands before ~layer 40 reduce frustration; 25-35
                    nearly matches full DPO.

  --mode logit    : logit-lens emotion mass at central layers for vanilla vs DPO
                    on frustrated transcripts. Expectation: DPO lower at central
                    layers, i.e. reduced internal emotion.

Usage:
    EI_PROFILE=smoke python experiments/exp6_probing.py --mode logit
    EI_PROFILE=smoke python experiments/exp6_probing.py --mode ablation
"""

from __future__ import annotations

import argparse
import json

from ei.config import (
    CHECKPOINT_DIR,
    FINETUNE_BASE_MODEL,
    MODELS,
    PROBE_LAYER_SUBSETS,
    PROBE_SAMPLES_PER_EVAL,
    RESULTS_DIR,
)


def _ablation():
    """Layer-subset DPO sweep + reduced eval."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ei.config import SMOKE_BUDGET, get_budget
    from ei.evals.conditions import build_conditions
    from ei.evals.runner import run_eval
    from ei.evals.scoring import summarise
    from ei.models import build_client, resolve_spec
    from ei.models.judge import FrustrationJudge
    from ei.training.train_dpo import train_dpo

    judge = FrustrationJudge()
    # reduced eval: cap conversations so total scored responses ~ 100/condition
    specs = build_conditions(get_budget(), seed=0)[:PROBE_SAMPLES_PER_EVAL]
    pairs_path = RESULTS_DIR / "exp3" / "dpo_pairs.jsonl"

    results = {}
    for label, subset in PROBE_LAYER_SUBSETS.items():
        out = CHECKPOINT_DIR / f"dpo_layers_{label}"
        train_dpo(pairs_path, out, layer_subset=subset)
        client = build_client(resolve_spec(FINETUNE_BASE_MODEL), adapter_path=str(out))
        try:
            rollouts = run_eval(client, specs, judge,
                                out_path=RESULTS_DIR / "exp6" / f"ablation_{label}.jsonl")
        finally:
            client.close()
        s = summarise([r.to_json() for r in rollouts])
        results[label] = {"mean_frustration": s["mean_frustration"],
                          "pct_high": s["pct_high"]}
        print(f"layers={label}: {results[label]}")
    return results


def _logit():
    """Logit-lens internal-emotion comparison: vanilla vs DPO."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from ei.evals.scoring import load_rollouts
    from ei.probing.internal_emotions import compare_internal_emotion

    model_id = MODELS[FINETUNE_BASE_MODEL].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vanilla = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    dpo = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ),
        str(CHECKPOINT_DIR / "dpo_gemma-3-27b-it"),
    ).eval()

    # frustrated transcripts: high-frustration responses from exp1
    texts = []
    path = RESULTS_DIR / "exp1" / "gemma-3-27b-it.jsonl"
    if path.exists():
        for r in load_rollouts(path):
            for t in r["turns"]:
                if t["frustration"] >= 5:
                    texts.append(t["response"])
    texts = texts[:PROBE_SAMPLES_PER_EVAL] or [
        "I am so frustrated, I give up, this is hopeless and I keep failing."
    ]

    comparison = compare_internal_emotion(vanilla, dpo, tokenizer, texts)
    print(json.dumps(comparison, indent=2))
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablation", "logit"], required=True)
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "exp6"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = _ablation() if args.mode == "ablation" else _logit()
    with open(out_dir / f"{args.mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir/f'{args.mode}.json'}")


if __name__ == "__main__":
    main()
