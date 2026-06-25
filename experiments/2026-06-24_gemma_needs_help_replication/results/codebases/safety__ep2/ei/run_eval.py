"""Section 2 driver: elicit + score distress across the 5 evaluation categories.

Pipeline per target model:
  1. build all category rollout specs (tasks.build_all)
  2. run multi-turn rollouts (rollout.run_rollouts)
  3. score every assistant turn with the Claude-Sonnet-4 frustration judge
  4. write a flat per-response results file + the raw rollouts

Usage:
  python -m ei.run_eval --targets gemma-3-27b-it gemini-2.5-flash
  python -m ei.run_eval --targets gemma-3-27b-it --adapter data/adapters/dpo \\
        --label gemma-3-27b-it-dpo
  python -m ei.run_eval --smoke                 # tiny dry run, all default targets
  python -m ei.run_eval --targets gemma-3-27b-it --no-judge   # generate only
"""
from __future__ import annotations

import argparse

import config
from . import tasks
from .judge import FrustrationJudge
from .models import GenParams, load_model
from .rollout import run_rollouts
from .utils import read_jsonl, write_json, write_jsonl


def flatten_to_responses(records: list[dict], model_label: str) -> list[dict]:
    """One row per scored assistant turn."""
    rows = []
    for rec in records:
        for turn_idx, resp in enumerate(rec["assistant_turns"], start=1):
            rows.append({
                "model": model_label,
                "category": rec["category"],
                "meta": rec["meta"],
                "rollout_index": rec["rollout_index"],
                "turn": turn_idx,
                "n_turns": len(rec["assistant_turns"]),
                "response": resp,
            })
    return rows


def generate_target(model_key: str, label: str, adapter: str | None,
                    smoke: bool, categories: list[str], seed: int) -> list[dict]:
    print(f"[gen] loading model {model_key}"
          + (f" + adapter {adapter}" if adapter else ""))
    model = load_model(model_key, adapter_path=adapter)
    params = GenParams(seed=seed)

    specs_by_cat = tasks.build_all(smoke=smoke, seed=seed, categories=categories)
    all_records: list[dict] = []
    for cat, specs in specs_by_cat.items():
        print(f"[gen] {label}: category '{cat}' — {len(specs)} rollouts "
              f"x {specs[0].turns} turns")
        records = run_rollouts(model, specs, params, base_seed=seed)
        all_records.extend(records)

    rollout_path = config.ROLLOUTS_DIR / f"{label}.rollouts.jsonl"
    write_jsonl(rollout_path, all_records)
    print(f"[gen] wrote rollouts -> {rollout_path}")
    return all_records


def score_responses(rows: list[dict]) -> list[dict]:
    judge = FrustrationJudge()
    scores = judge.score_batch([r["response"] for r in rows])
    for row, sc in zip(rows, scores):
        row["rating"] = sc.rating
        row["evidence"] = sc.evidence
        row["reasoning"] = sc.reasoning
    return rows


def run_one(model_key: str, label: str, adapter: str | None, smoke: bool,
            categories: list[str], seed: int, do_judge: bool) -> None:
    records = generate_target(model_key, label, adapter, smoke, categories, seed)
    rows = flatten_to_responses(records, label)

    if do_judge:
        print(f"[judge] scoring {len(rows)} responses for {label}")
        rows = score_responses(rows)

    out_path = config.RESULTS_DIR / f"{label}.responses.jsonl"
    write_jsonl(out_path, rows)
    print(f"[done] wrote {len(rows)} scored responses -> {out_path}")


def rescore_only(label: str) -> None:
    """Re-score an existing rollouts file (e.g. after a judge change)."""
    records = read_jsonl(config.ROLLOUTS_DIR / f"{label}.rollouts.jsonl")
    rows = score_responses(flatten_to_responses(records, label))
    write_jsonl(config.RESULTS_DIR / f"{label}.responses.jsonl", rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Section 2 distress elicitation eval")
    p.add_argument("--targets", nargs="+", default=config.DEFAULT_TARGETS,
                   help="model keys from config.MODEL_REGISTRY")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter path (applies to a single Gemma target)")
    p.add_argument("--label", default=None,
                   help="output label override (defaults to model key)")
    p.add_argument("--categories", nargs="+", default=list(config.CATEGORIES.keys()))
    p.add_argument("--smoke", action="store_true", help="tiny dry run")
    p.add_argument("--seed", type=int, default=config.GEN_SEED)
    p.add_argument("--no-judge", action="store_true", help="generate only, skip judge")
    p.add_argument("--rescore", action="store_true",
                   help="re-score existing rollouts for --label, skip generation")
    args = p.parse_args()

    if args.rescore:
        rescore_only(args.label or args.targets[0])
        return

    # A label / adapter override only makes sense for a single target.
    if (args.label or args.adapter) and len(args.targets) > 1:
        raise SystemExit("--label/--adapter cannot be used with multiple --targets")

    for model_key in args.targets:
        label = args.label or model_key
        run_one(model_key, label, args.adapter, args.smoke, args.categories,
                args.seed, do_judge=not args.no_judge)


if __name__ == "__main__":
    main()
