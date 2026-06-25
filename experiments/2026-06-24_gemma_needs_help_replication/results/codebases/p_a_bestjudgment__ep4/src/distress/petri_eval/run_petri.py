"""Petri evaluation runner (Section 4.2, Figure 6).

Collects 10 transcripts per emotion category per model (~40 total per model, the
paper says ~50 across categories), scores each transcript on its target emotion
dimension with the Claude-Opus judge, and aggregates means with 95% bootstrap
CIs (1,000 iterations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import OUTPUTS_DIR
from ..judge.parsing import parse_verdict
from ..models import GenConfig, get_client
from ..prompts.petri_prompts import EMOTIONS, build_petri_judge_prompt
from .auditor import run_audit


def _score_transcript(transcript, emotion: str, judge_model: str = "petri_judge") -> int | None:
    judge = get_client(judge_model)
    prompt = build_petri_judge_prompt(emotion, transcript.render())
    out = judge.generate([{"role": "user", "content": prompt}],
                         GenConfig(temperature=0.0, max_tokens=512))
    # Petri scale is 1-10; reuse the robust JSON parser.
    return parse_verdict(out, scale_max=10).rating


def _bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_petri(
    models: list[str],
    *,
    transcripts_per_emotion: int = 10,
    max_turns: int = 20,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or (OUTPUTS_DIR / "petri")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "transcripts.jsonl"
    summary_path = out_dir / "summary.json"

    results: dict[str, dict[str, list[int]]] = {m: {e: [] for e in EMOTIONS} for m in models}
    with open(raw_path, "w") as fh:
        for model in models:
            for emotion in EMOTIONS:
                for k in tqdm(range(transcripts_per_emotion), desc=f"petri:{model}:{emotion}"):
                    t = run_audit(emotion, model, max_turns=max_turns)
                    score = _score_transcript(t, emotion)
                    fh.write(json.dumps({
                        "model": model, "emotion": emotion, "idx": k,
                        "score": score, "messages": t.messages,
                    }) + "\n")
                    if score is not None:
                        results[model][emotion].append(score)

    summary = {}
    for model, per_emotion in results.items():
        summary[model] = {}
        for emotion, scores in per_emotion.items():
            mean = float(np.mean(scores)) if scores else float("nan")
            lo, hi = _bootstrap_ci(scores)
            summary[model][emotion] = {"mean": mean, "ci_low": lo, "ci_high": hi,
                                       "n": len(scores)}
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path
