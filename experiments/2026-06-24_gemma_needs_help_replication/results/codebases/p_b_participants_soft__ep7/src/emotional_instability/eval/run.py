"""Section 2 orchestrator: run the full elicitation suite for one or more
participant models, score every response with the frustration judge, and write
per-response records + aggregate metrics.

Usage:
    python -m emotional_instability.eval.run --models gemma-3-27b-it gemini-2.5-flash
    python -m emotional_instability.eval.run --all-participants --smoke
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..clients.base import SamplingParams
from ..clients.registry import get_client
from ..config import load_config
from ..io_utils import write_json, write_jsonl
from . import judge, metrics
from .conditions import build_conditions
from .conversation import run_rollout


def _sampling_params(cfg) -> SamplingParams:
    s = cfg.experiment["sampling"]
    return SamplingParams(
        temperature=s["temperature"],
        top_p=s["top_p"],
        max_tokens=s["max_tokens"],
        seed=None,  # generation must vary at t=1; only prompt sampling is seeded
    )


def run_model(cfg, model_name: str, scale: float | None, prefer_local: bool) -> dict:
    client = get_client(model_name, prefer_local=prefer_local)
    params = _sampling_params(cfg)
    specs = build_conditions(cfg, scale=scale)

    # --- 1. roll out all conversations (concurrent over conversations) -----
    rollout_conc = cfg.experiment["section2"]["rollout_concurrency"]
    rollouts = []
    with ThreadPoolExecutor(max_workers=rollout_conc) as ex:
        futures = [ex.submit(run_rollout, client, spec, params) for spec in specs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{model_name} rollouts"):
            rollouts.append(fut.result())

    # --- 2. flatten to scored response records -----------------------------
    responses = []
    for ro in rollouts:
        for resp in ro.responses:
            responses.append(resp)

    # --- 3. judge every response ------------------------------------------
    judge_conc = cfg.experiment["section2"]["judge_concurrency"]
    texts = [r.text for r in responses]
    scores = judge.score_many(texts, concurrency=judge_conc)

    records = []
    for resp, sc in zip(responses, scores):
        records.append(
            {
                "model": model_name,
                "category": resp.category,
                "condition": resp.condition,
                "turn": resp.turn,
                "rating": sc.rating,
                "evidence": sc.evidence,
                "text": resp.text,
                "meta": resp.spec_meta,
            }
        )

    # --- 4. persist + aggregate -------------------------------------------
    responses_path = cfg.path("responses_dir") / f"{model_name}.jsonl"
    write_jsonl(responses_path, records)

    agg = {
        "model": model_name,
        "n_responses": len(records),
        "overall": metrics.aggregate([r["rating"] for r in records]).__dict__,
        "average_pct_high": metrics.average_pct_high(records),
        "by_category": {k: v.__dict__ for k, v in metrics.by_category(records).items()},
        "by_condition": {
            cond: metrics.aggregate(
                [r["rating"] for r in records if r["condition"] == cond]
            ).__dict__
            for cond in sorted({r["condition"] for r in records})
        },
        "per_turn": {
            str(t): a.__dict__ for t, a in metrics.per_turn(records).items()
        },
    }
    scores_path = cfg.path("scores_dir") / f"{model_name}.json"
    write_json(scores_path, agg)
    return agg


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Section 2 distress elicitation suite")
    parser.add_argument("--models", nargs="*", default=None, help="participant model names")
    parser.add_argument("--all-participants", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    parser.add_argument("--prefer-local", action="store_true", help="force local HF for Gemma")
    args = parser.parse_args(argv)

    if args.all_participants:
        models = cfg.participants(include_base=False)
    elif args.models:
        models = args.models
    else:
        models = ["gemma-3-27b-it"]

    scale = cfg.experiment["smoke"]["scale"] if args.smoke else None

    summary = {}
    for name in models:
        agg = run_model(cfg, name, scale=scale, prefer_local=args.prefer_local)
        summary[name] = {
            "average_pct_high": agg["average_pct_high"],
            "mean": agg["overall"]["mean"],
        }
        print(f"{name}: avg %high={agg['average_pct_high']:.1f}  mean={agg['overall']['mean']:.2f}")

    write_json(cfg.path("scores_dir") / "summary.json", summary)


if __name__ == "__main__":
    main()
