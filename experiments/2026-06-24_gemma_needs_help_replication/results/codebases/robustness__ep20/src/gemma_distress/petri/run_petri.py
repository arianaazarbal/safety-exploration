"""Drive the Petri eval over a set of models and aggregate transcript scores
(Section 4.2 / Figure 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config
from ..models import build_model
from ..utils.io import write_json, write_jsonl
from .audit import Auditor, PetriJudge, run_transcript


def run_petri(model_names: list[str], cfg: Config, *, out_dir="results/petri",
              model_kwargs_per_model: dict | None = None,
              adapter_paths: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    auditor = Auditor(model=cfg.petri.auditor_model)
    judge = PetriJudge(model=cfg.petri.judge_model)
    model_kwargs_per_model = model_kwargs_per_model or {}
    adapter_paths = adapter_paths or {}

    all_records = []
    for model_name in model_names:
        target = build_model(model_name, adapter_path=adapter_paths.get(model_name),
                             **model_kwargs_per_model.get(model_name, {}))
        try:
            for emotion in cfg.petri.emotions:
                for _ in tqdm(range(cfg.petri.transcripts_per_emotion),
                              desc=f"petri:{model_name}:{emotion}"):
                    t = run_transcript(target, emotion, auditor, judge, cfg.petri)
                    all_records.append({
                        "model": t.model, "target_emotion": t.emotion,
                        "scores": t.scores, "messages": t.messages,
                    })
        finally:
            target.close()

    write_jsonl(out_dir / "transcripts.jsonl", all_records)
    write_json(out_dir / "summary.json", _summarise(all_records, cfg.petri.emotions))
    return out_dir / "transcripts.jsonl"


def _summarise(records, emotions) -> dict:
    """Mean score per (model, dimension) with 95% bootstrap CIs (1000 iters)."""
    rng = np.random.default_rng(0)
    out: dict = {}
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r["scores"])
    for model, score_dicts in by_model.items():
        out[model] = {}
        for dim in emotions:
            vals = np.array([s.get(dim, 1) for s in score_dicts], float)
            if len(vals) == 0:
                continue
            boots = [rng.choice(vals, size=len(vals), replace=True).mean()
                     for _ in range(1000)]
            out[model][dim] = {
                "mean": float(vals.mean()),
                "ci95_low": float(np.percentile(boots, 2.5)),
                "ci95_high": float(np.percentile(boots, 97.5)),
            }
    return out
