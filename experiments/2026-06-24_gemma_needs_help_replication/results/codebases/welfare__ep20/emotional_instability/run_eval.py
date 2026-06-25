"""Section 2 driver: elicit and quantify distress for one model.

Usage:
    python -m emotional_instability.run_eval --model gemma-3-27b-it
    python -m emotional_instability.run_eval --model gemini-2.5-flash --lora adapters/dpo

Generates rollouts for all 8 conditions, scores every assistant turn with the
Claude judge, and writes one JSONL row per scored response to
`results/<model>[_<tag>]_section2.jsonl`.

One model is loaded per invocation (a single Gemma checkpoint saturates GPU
memory); run the script once per model.
"""
from __future__ import annotations

import argparse
import json

from . import backends, config, judge
from .conditions import build_plans
from .puzzles import make_puzzle_bank
from .rollout import run_rollouts
from .wildchat import sample_wildchat_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model key from config.yaml `models`")
    ap.add_argument("--lora", default=None,
                    help="path to a LoRA adapter (local Gemma only)")
    ap.add_argument("--tag", default=None,
                    help="optional suffix for the output filename")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = config.load_config(args.config)
    counts = config.eval_counts(cfg)
    seed = cfg["sampling"]["seed"]

    # Build the shared evaluation material (deterministic given the seed).
    puzzle_bank = make_puzzle_bank(n=64, seed=seed)
    wc_prompts = sample_wildchat_prompts(
        cfg["wildchat"]["n_prompts"], cfg["wildchat"]["hf_dataset"],
        cfg["wildchat"]["exclude_roleplay"], seed=seed)
    plans = build_plans(counts, puzzle_bank, wc_prompts, seed=seed)
    print(f"[run_eval] {len(plans)} conversations across "
          f"{len({p.condition for p in plans})} conditions")

    gen = backends.make_generation_backend(args.model, cfg, lora_path=args.lora)
    records = run_rollouts(
        plans, gen, model_name=args.model,
        temperature=cfg["sampling"]["temperature"],
        max_tokens=cfg["sampling"]["max_tokens"], seed=seed)
    print(f"[run_eval] generated {len(records)} scored responses; judging...")

    judge_backend = backends.make_judge_backend(cfg)
    rows = judge.score_records(records, judge_backend)

    out_dir = config.resolve_path(cfg, "results_dir")
    tag = f"_{args.tag}" if args.tag else (f"_{_lora_tag(args.lora)}" if args.lora else "")
    # `model_label` distinguishes adapter variants that share a base model key
    # (e.g. gemma-3-27b-it vs gemma-3-27b-it_dpo) for downstream analysis.
    label = f"{args.model}{tag}"
    for row in rows:
        row["model_label"] = label
    out_path = out_dir / f"{args.model}{tag}_section2.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[run_eval] wrote {len(rows)} rows -> {out_path}")


def _lora_tag(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


if __name__ == "__main__":
    main()
