"""Judge-reliability report (Section 2.1): Pearson r and within-one-point rate
between the primary Claude judge and the GPT-5-mini cross-check.

Thin wrapper over judge.cross_check that pulls responses + primary ratings from a
saved elicitation records.jsonl.
"""
from __future__ import annotations

from dataclasses import asdict

from ..config import Config, load_config
from ..judge.cross_check import cross_check_agreement
from ..utils.io import read_jsonl, write_json


def compute_agreement(cfg: Config | None = None, model_name: str | None = None) -> dict:
    cfg = cfg or load_config()
    model_name = model_name or cfg.elicitation_models[0]
    records = list(read_jsonl(
        cfg.output_root() / "elicitation" / model_name / "records.jsonl"
    ))
    responses = [r["response_text"] for r in records if r.get("rating") is not None]
    ratings = [r["rating"] for r in records if r.get("rating") is not None]

    report = cross_check_agreement(
        responses, ratings,
        n_samples=cfg.cross_check.n_samples,
        provider=cfg.cross_check.provider,
        model=cfg.cross_check.model,
        seed=cfg.seed,
    )
    out = asdict(report)
    write_json(cfg.output_root() / "elicitation" / "judge_agreement.json", out)
    return out
