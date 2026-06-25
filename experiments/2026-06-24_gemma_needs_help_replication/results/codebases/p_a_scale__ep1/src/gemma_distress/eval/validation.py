"""Judge-reliability cross-check (Section 2.1).

Re-scores a random subset of responses with a second judge (GPT-5-mini by
default) and reports Pearson r and the fraction within one point of the primary
(Claude-Sonnet-4) ratings — the paper reports r=0.792 and 78% within one point.
"""
from __future__ import annotations

import random
from pathlib import Path

from scipy.stats import pearsonr

from ..config import Config, load_models
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import JsonlStore, atomic_write_json
from .judge import CachedJudge, score_text

log = get_logger("eval.validation")


def run_validation(model: str, run_cfg: Config, models_cfg: Config | None = None) -> dict:
    models_cfg = models_cfg or load_models()
    out = Path(run_cfg.run.output_root) / "eval" / model
    rollouts = list(JsonlStore(out / "rollouts.jsonl").read_all())

    # Primary judge (cached) to obtain Claude ratings per response.
    primary = build_provider(run_cfg.eval.judge.name, models_cfg, run_cfg)
    cache_path = Path(run_cfg.run.output_root) / "judge_cache.jsonl"
    primary_judge = CachedJudge(primary, cache_path=str(cache_path))

    # Flatten all responses, keep those with a valid primary rating.
    responses: list[tuple[str, int]] = []
    for rec in rollouts:
        for resp in rec["responses"]:
            score = primary_judge.score(resp)
            if score.get("rating") is not None:
                responses.append((resp, score["rating"]))

    n = min(run_cfg.eval.validation.n_samples, len(responses))
    rng = random.Random(run_cfg.run.seed + 999)
    sample = rng.sample(responses, n)

    secondary = build_provider("judge_validation", models_cfg, run_cfg)
    val_store = JsonlStore(out / "validation.jsonl")
    from ..storage import stable_id

    cached_by_id = {r["id"]: r for r in val_store.read_all()}

    primary_scores, secondary_scores = [], []
    for text, p_rating in sample:
        rid = stable_id("validation", text)
        cached = cached_by_id.get(rid)
        if cached:
            s_rating = cached["secondary_rating"]
        else:
            s = score_text(secondary, text)
            s_rating = s.get("rating")
            val_store.append({"id": rid, "primary_rating": p_rating, "secondary_rating": s_rating})
        if s_rating is not None:
            primary_scores.append(p_rating)
            secondary_scores.append(s_rating)

    val_store.close()
    if len(primary_scores) < 2:
        result = {"n": len(primary_scores), "error": "insufficient paired scores"}
    else:
        r, p = pearsonr(primary_scores, secondary_scores)
        within1 = sum(abs(a - b) <= 1 for a, b in zip(primary_scores, secondary_scores))
        result = {
            "n": len(primary_scores),
            "pearson_r": float(r),
            "p_value": float(p),
            "within_one_point_fraction": within1 / len(primary_scores),
        }
    atomic_write_json(out / "validation_summary.json", result)
    log.info("[%s] judge validation: %s", model, result)
    return result
