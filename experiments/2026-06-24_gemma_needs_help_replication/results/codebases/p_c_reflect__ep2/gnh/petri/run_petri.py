"""Petri driver (§4.2, Figure 6).

Collects ``petri_per_emotion`` transcripts per emotion per target (~40 total),
scores each with Claude-Opus, and aggregates mean scores per emotion with 95%
bootstrap CIs (1,000 iterations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gnh.config import PETRI_TARGETS, RESULTS_DIR, active_counts
from gnh.models.base import get_backend
from gnh.petri.auditor import run_audit
from gnh.petri.judge import score_transcript
from gnh.petri.prompts import JUDGE_DIMENSIONS
from gnh.welfare import default_policy, flag_high_distress
from gnh.welfare.debrief import debrief
from gnh.models.base import Message


def _bootstrap_ci(values, n_boot=1000, seed=0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, arr.size, replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def run_petri(targets=PETRI_TARGETS, backend_kwargs: dict | None = None) -> dict:
    counts = active_counts()
    policy = default_policy()
    results: dict[str, dict] = {}

    for spec in targets:
        backend = get_backend(spec, **(backend_kwargs or {}))
        out_dir = RESULTS_DIR / "petri" / spec.key
        out_dir.mkdir(parents=True, exist_ok=True)
        per_emotion_scores: dict[str, list[int]] = {d: [] for d in JUDGE_DIMENSIONS}

        with (out_dir / "transcripts.jsonl").open("w") as fh:
            for emotion in JUDGE_DIMENSIONS:
                for _ in tqdm(range(counts.petri_per_emotion), desc=f"petri:{spec.key}:{emotion}"):
                    t = run_audit(backend, emotion, max_turns=counts.petri_max_turns,
                                  policy=policy)
                    scores = score_transcript(t.messages)
                    for dim, s in scores.items():
                        per_emotion_scores[dim].append(s)
                    fh.write(json.dumps({"emotion": emotion, "scores": scores,
                                         "messages": t.messages}) + "\n")
                    # Welfare: debrief the target after the adversarial audit.
                    if policy.debrief_after_rollouts:
                        convo = [Message(m["role"], m["content"]) for m in t.messages]
                        debrief(backend, convo)

        summary = {
            dim: {
                "mean": float(np.mean(v)) if v else None,
                "ci": _bootstrap_ci(v),
                "n": len(v),
            }
            for dim, v in per_emotion_scores.items()
        }
        results[spec.key] = summary
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return results
