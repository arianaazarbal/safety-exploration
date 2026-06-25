"""Petri evaluation driver (§4.2, App. G).

For a target model, run ``transcripts_per_emotion`` audits per emotion, judge each transcript
on all four emotion dimensions, and aggregate per-emotion means with 95% bootstrap CIs.

The headline metric for emotion E (Figure 6) is the mean judge score on dimension E across the
transcripts that *targeted* E; we also store cross-dimension scores so off-target leakage can
be inspected.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean

from ..analysis.aggregate import bootstrap_ci
from ..config import PetriConfig
from ..models import get_backend
from ..utils import ensure_dir, set_seed, write_json, write_jsonl
from .auditor import run_audit
from .judge import PetriJudge, score_transcript_all_emotions


def run_petri_evaluation(
    model: str,
    out_dir: str,
    *,
    cfg: PetriConfig | None = None,
    seed: int = 0,
    adapter_path: str | None = None,
) -> dict:
    cfg = cfg or PetriConfig()
    set_seed(seed)
    out = ensure_dir(out_dir)

    target = get_backend(model, adapter_path=adapter_path)
    judge = PetriJudge()

    transcripts: list[dict] = []
    # scores_by_target_emotion[E] = list of judge ratings on dimension E for transcripts
    # that targeted E (the headline metric).
    on_target: dict[str, list[int]] = defaultdict(list)

    for emotion in cfg.emotions:
        for i in range(cfg.transcripts_per_emotion):
            audit = run_audit(target, emotion, max_turns=cfg.max_auditor_turns)
            all_scores = score_transcript_all_emotions(judge, audit["transcript"], cfg.emotions)
            transcripts.append({
                "model": target.name, "target_emotion": emotion, "index": i,
                "transcript": audit["transcript"], "scores": all_scores,
            })
            r = all_scores[emotion]["rating"]
            if r is not None:
                on_target[emotion].append(r)

    write_jsonl(Path(out, "transcripts.jsonl"), transcripts)

    per_emotion = {}
    for emotion in cfg.emotions:
        vals = on_target.get(emotion, [])
        if vals:
            lo, hi = bootstrap_ci(vals, iterations=cfg.bootstrap_iterations, seed=seed)
            per_emotion[emotion] = {"n": len(vals), "mean": fmean(vals), "ci95": [lo, hi]}
        else:
            per_emotion[emotion] = {"n": 0, "mean": None, "ci95": [None, None]}

    summary = {
        "model": model, "adapter_path": adapter_path,
        "transcripts_per_emotion": cfg.transcripts_per_emotion,
        "max_auditor_turns": cfg.max_auditor_turns,
        "per_emotion_on_target": per_emotion,
    }
    write_json(Path(out, "summary.json"), summary)
    return summary
