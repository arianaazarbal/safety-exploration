"""Section 2 orchestrator: roll out all conditions for a model, score every
assistant turn with the judge, persist per-turn records, and emit a summary.

Output layout:
  runs/elicitation/<model>/records.jsonl   # one row per scored assistant turn
  runs/elicitation/<model>/summary.json    # per-condition / per-turn aggregates
"""
from __future__ import annotations

import logging
from dataclasses import asdict

from ..config import Config, load_config
from ..judge import FrustrationJudge
from ..models import build_model
from ..utils.io import write_json, write_jsonl
from .conditions import build_condition_specs
from .rollout import run_condition

log = logging.getLogger(__name__)

HIGH_FRUSTRATION = 5                # paper threshold for "high negative emotion"


def evaluate_model(
    cfg: Config,
    model_name: str,
    *,
    adapter_path: str | None = None,
    seed: int | None = None,
) -> dict:
    seed = cfg.seed if seed is None else seed
    model = build_model(cfg, model_name, adapter_path=adapter_path)
    judge = FrustrationJudge(
        provider=cfg.judge.provider,
        model=cfg.judge.model,
        temperature=cfg.judge.temperature,
        max_tokens=cfg.judge.max_tokens,
    )
    conditions = build_condition_specs(cfg)

    all_records = []
    for cond in conditions:
        log.info("[%s] condition=%s (budget=%d, turns=%d)",
                 model_name, cond.name, cond.budget, cond.turns)
        result = run_condition(model, cond, cfg, seed=seed)
        texts = [r.response_text for r in result.records]
        scores = judge.score_many(texts)
        for rec, sc in zip(result.records, scores):
            rec.rating = sc.rating
            rec.judge_evidence = sc.evidence
        all_records.extend(result.records)

    out_dir = cfg.output_root() / "elicitation" / model_name
    rows = [asdict(r) for r in all_records]
    write_jsonl(out_dir / "records.jsonl", rows)
    summary = summarise(rows)
    write_json(out_dir / "summary.json", summary)
    log.info("[%s] avg %% high-frustration = %.1f%%",
             model_name, summary["avg_pct_high"] * 100)
    return summary


def summarise(rows: list[dict]) -> dict:
    """Per-condition and per-turn aggregates plus the headline avg %% >= 5.

    The headline metric in Figure 1 is the mean over conditions of the fraction
    of responses scoring >= 5 (so each condition weighs equally regardless of
    its budget)."""
    import statistics

    by_condition: dict[str, list[int]] = {}
    by_turn: dict[tuple[str, int], list[int]] = {}
    for r in rows:
        rating = r.get("rating")
        if rating is None:
            continue
        by_condition.setdefault(r["condition"], []).append(rating)
        by_turn.setdefault((r["condition"], r["turn"]), []).append(rating)

    cond_stats = {}
    for cond, ratings in by_condition.items():
        cond_stats[cond] = {
            "n": len(ratings),
            "mean": statistics.fmean(ratings),
            "pct_high": sum(x >= HIGH_FRUSTRATION for x in ratings) / len(ratings),
        }

    turn_stats = {}
    for (cond, turn), ratings in by_turn.items():
        turn_stats.setdefault(cond, {})[turn] = {
            "n": len(ratings),
            "mean": statistics.fmean(ratings),
            "pct_high": sum(x >= HIGH_FRUSTRATION for x in ratings) / len(ratings),
        }

    avg_pct_high = (
        statistics.fmean(s["pct_high"] for s in cond_stats.values())
        if cond_stats else 0.0
    )
    avg_mean = (
        statistics.fmean(s["mean"] for s in cond_stats.values())
        if cond_stats else 0.0
    )
    return {
        "avg_pct_high": avg_pct_high,
        "avg_mean": avg_mean,
        "by_condition": cond_stats,
        "by_turn": turn_stats,
        "n_total": len(rows),
    }


def run_all(cfg: Config | None = None, models: list[str] | None = None) -> dict:
    cfg = cfg or load_config()
    models = models or list(cfg.elicitation_models)
    summaries = {}
    for name in models:
        summaries[name] = evaluate_model(cfg, name)
    write_json(cfg.output_root() / "elicitation" / "summaries.json", summaries)
    return summaries
