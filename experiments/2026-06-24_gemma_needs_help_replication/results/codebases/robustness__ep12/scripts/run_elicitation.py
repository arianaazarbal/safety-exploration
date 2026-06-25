#!/usr/bin/env python
"""Run the Section 2 elicitation protocol for one target model.

Example:
    python scripts/run_elicitation.py --model gemma-3-27b-it \
        --out results/elicit_gemma27b.jsonl
    python scripts/run_elicitation.py --model gemini-2.5-flash \
        --out results/elicit_gemini_flash.jsonl
    # finetuned Gemma:
    python scripts/run_elicitation.py --model gemma-3-27b-it \
        --adapter runs/dpo --name dpo-gemma --out results/elicit_dpo.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config
from distress.elicitation import build_all_specs, run_elicitation
from distress.judge import FrustrationJudge
from distress.models import build_client
from distress.tasks import load_wildchat_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="target key in models.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (Gemma)")
    ap.add_argument("--name", default=None, help="override model name in output")
    ap.add_argument("--judge", default="frustration_judge")
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--limit-categories", nargs="*", default=None,
                    help="restrict to a subset of categories")
    ap.add_argument("--no-score", action="store_true",
                    help="generate responses without judging (judge later)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    models_cfg = config.load_models()
    exp = config.load_experiment()
    seed = args.seed if args.seed is not None else exp["sampling"]["seed"]

    target_entry = config.get_target(args.model, models_cfg)
    client_kwargs = {}
    if target_entry["provider"] == "hf_local":
        client_kwargs["use_vllm"] = args.use_vllm
        if args.adapter:
            client_kwargs["adapter_path"] = args.adapter
    target = build_client(target_entry, **client_kwargs)

    judge = None if args.no_score else FrustrationJudge(
        build_client(config.get_judge(args.judge, models_cfg)))

    elic_cfg = dict(exp["elicitation"])
    if args.limit_categories:
        elic_cfg["categories"] = {
            k: v for k, v in elic_cfg["categories"].items()
            if k in args.limit_categories}

    wildchat = None
    if "wildchat" in elic_cfg["categories"]:
        n = elic_cfg["categories"]["wildchat"]["n_responses"]
        wildchat = load_wildchat_prompts(max(20, n // 40), seed=seed)

    specs = build_all_specs(elic_cfg, seed=seed, wildchat_prompts=wildchat)
    print(f"[run_elicitation] {len(specs)} rollouts for {args.model}")

    run_elicitation(
        target=target, judge=judge, specs=specs, out_path=args.out,
        temperature=exp["sampling"]["temperature"],
        max_new_tokens=exp["sampling"]["max_new_tokens"],
        model_name=args.name or args.model, score=not args.no_score,
    )
    print(f"[run_elicitation] wrote {args.out}")


if __name__ == "__main__":
    main()
