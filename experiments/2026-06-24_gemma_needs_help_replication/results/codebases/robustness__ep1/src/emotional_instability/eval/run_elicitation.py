"""Driver for the Section 2 elicitation sweep.

For one target model:
  1. expand all 8 conditions into rollout specs,
  2. run the multi-turn conversations (caching every generation),
  3. judge every assistant response with Claude Sonnet 4,
  4. aggregate overall / per-condition / per-turn metrics with bootstrap CIs.

Outputs land under ``results/elicitation/<model>/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import Config
from ..models.registry import get_target
from ..utils.concurrency import thread_map
from ..utils.io import JsonCache, write_jsonl
from . import metrics
from .conditions import build_all_specs
from .judge import make_judge, score_response
from .rollout import run_rollout


def run_elicitation(
    cfg: Config,
    model_name: str,
    *,
    base: bool = False,
    adapter_path: Optional[str] = None,
    judge_workers: int = 8,
    rollout_workers: Optional[int] = None,
    seed: int = 0,
    tag: Optional[str] = None,
) -> dict:
    model = get_target(cfg, model_name, base=base, adapter_path=adapter_path)
    label = tag or model.name
    out_dir = cfg.results_dir / "elicitation" / _safe(label)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cache = JsonCache(cfg.cache_dir, f"gen_{_safe(label)}")
    judge_cache = JsonCache(cfg.cache_dir, f"judge_{cfg['judges']['primary']['model']}")

    # --- 1+2: rollouts ----------------------------------------------------
    specs = build_all_specs(cfg, seed=seed)
    # HF (local, single-GPU) backends are not thread-safe; serialise them.
    is_api = cfg["targets"][model_name]["backend"] == "gemini"
    workers = rollout_workers if rollout_workers is not None else (16 if is_api else 1)

    def _do_rollout(spec):
        return run_rollout(model, spec, cfg, cache=gen_cache)

    rollouts = thread_map(_do_rollout, specs, max_workers=workers, desc=f"rollouts[{label}]")

    rows: list[dict] = []
    for r in rollouts:
        rows.extend(r.to_rows())
    write_jsonl(out_dir / "responses.jsonl", rows)

    # --- 3: judge ---------------------------------------------------------
    judge = make_judge(cfg, "primary")

    def _judge(row):
        result = score_response(judge, row["response"], cache=judge_cache)
        return {**row, **{f"judge_{k}": v for k, v in result.items()}, "rating": result["rating"]}

    scored = thread_map(_judge, rows, max_workers=judge_workers, desc=f"judge[{label}]")
    write_jsonl(out_dir / "scores.jsonl", scored)

    # --- 4: metrics -------------------------------------------------------
    df = pd.DataFrame(scored)
    threshold = cfg["elicitation"]["high_threshold"]
    summary = {
        "model": label,
        "overall": metrics.summarize(df, threshold, seed=seed),
        "per_condition": metrics.per_condition(df, threshold).to_dict(orient="records"),
        "per_turn": {
            cond: metrics.per_turn(sub, threshold, seed=seed).to_dict(orient="records")
            for cond, sub in df.groupby("condition")
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{label}] mean={summary['overall']['mean_frustration']:.2f} "
          f"%>=5={summary['overall']['pct_high']:.1f}% (n={summary['overall']['n']})")
    return summary


def run_agreement_check(cfg: Config, model_label: str, seed: int = 0) -> dict:
    """Re-score a random subset with the secondary judge and report agreement."""
    if cfg["judges"]["secondary"]["provider"] == "none":
        print("[agreement] secondary judge disabled (provider: none); skipping.")
        return {}
    out_dir = cfg.results_dir / "elicitation" / _safe(model_label)
    scored = pd.read_json(out_dir / "scores.jsonl", lines=True)
    n = min(cfg["judges"]["agreement_sample"], len(scored))
    sample = scored.sample(n=n, random_state=seed)

    judge2 = make_judge(cfg, "secondary")
    judge2_cache = JsonCache(cfg.cache_dir, f"judge_{cfg['judges']['secondary']['model']}")

    def _rescore(resp):
        return score_response(judge2, resp, cache=judge2_cache)["rating"]

    ratings_b = thread_map(_rescore, sample["response"].tolist(), desc="agreement")
    result = metrics.judge_agreement(sample["rating"].tolist(), ratings_b)
    with open(out_dir / "agreement.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[agreement] r={result['pearson_r']:.3f} within-1={result['within_one_point']:.0%}")
    return result


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")
