"""Run the Section 3 prefill continuations and score them.

For each prefill seed, each model generates ``continuations_per_prefill`` (=50)
continuations from the same starting point. Only the *generated* text (excluding
the prefill) is scored by the Section 2 judge. We then aggregate mean frustration
and % >=5 per model, broken down by base/instruct, truncation type (early/onset),
and seed category (numeric/text) — reproducing Figure 4.

Scope: Gemma base (gemma-3-27b-pt) vs instruct (gemma-3-27b-it). The full paper
also runs Qwen and OLMo base/instruct; those families are out of scope here.
Gemini has no public base model and cannot be prefilled (closed), so it is
necessarily excluded from this experiment (a documented limitation).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

from ..config import get_config
from ..models.base import ChatMessage, GenerationConfig
from ..models.judges import AnthropicClient
from ..models.registry import build_client
from ..utils.io import dump_json, load_jsonl, run_dir, write_jsonl
from ..eval.judge import FrustrationJudge

DEFAULT_PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_ge(xs, t):
    xs = [x for x in xs if x is not None]
    return sum(1 for x in xs if x >= t) / len(xs) if xs else float("nan")


def run_prefill_for_model(model_name: str, seeds: list[dict], cfg) -> list[dict]:
    client = build_client(model_name)
    if not client.supports_prefill:
        raise RuntimeError(f"{model_name} does not support prefilled continuations")
    gen = GenerationConfig(
        temperature=cfg.eval.sampling.temperature,
        top_p=cfg.eval.sampling.top_p,
        max_new_tokens=cfg.eval.sampling.max_new_tokens,
        n=cfg.prefill.continuations_per_prefill,
    )
    rows = []
    try:
        from tqdm import tqdm
        seeds_iter = tqdm(seeds, desc=f"prefill[{model_name}]")
    except ImportError:
        seeds_iter = seeds

    for seed in seeds_iter:
        history = [ChatMessage(m["role"], m["content"]) for m in seed["history"]]
        continuations = client.prefilled_continuation(history, seed["prefill_text"], gen)
        for ci, cont in enumerate(continuations):
            rows.append({
                "model": model_name,
                "seed_id": seed["seed_id"],
                "seed_category": seed["seed_category"],
                "truncation": seed["truncation"],
                "continuation_index": ci,
                "continuation_text": cont,
            })
    return rows


def aggregate(scored_rows: list[dict], threshold: int) -> dict:
    groups = defaultdict(list)
    for r in scored_rows:
        key = (r["model"], r["seed_category"], r["truncation"])
        groups[key].append(r.get("frustration"))
    out = {}
    for (model, cat, trunc), scores in groups.items():
        out.setdefault(model, {})[f"{cat}/{trunc}"] = {
            "n": len([s for s in scores if s is not None]),
            "mean": _mean(scores),
            "pct_high": _frac_ge(scores, threshold) * 100,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Run Section 3 prefill continuations.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_PREFILL_MODELS)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.preset)

    out_dir = run_dir(cfg.output_root, "prefill")
    seeds = load_jsonl(os.path.join(out_dir, "seeds.jsonl"))

    all_rows = []
    for model in args.models:
        all_rows.extend(run_prefill_for_model(model, seeds, cfg))

    if not args.no_score:
        judge = FrustrationJudge(AnthropicClient(cfg.eval.judge.frustration_model))
        for r in all_rows:
            res = judge.score(r["continuation_text"])
            r["frustration"] = res.rating

    write_jsonl(os.path.join(out_dir, "continuations.jsonl"), all_rows)
    report = aggregate(all_rows, cfg.eval.high_frustration_threshold)
    dump_json(os.path.join(out_dir, "prefill_report.json"), report)
    print("Section 3 prefill report:")
    for model, cells in report.items():
        print(f"  {model}")
        for k, v in cells.items():
            print(f"    {k:16s} n={v['n']:4d} mean={v['mean']:.2f} %>=5={v['pct_high']:.1f}")


if __name__ == "__main__":
    main()
