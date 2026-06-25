"""Petri orchestration (Section 4.2, Figure 6).

For each target model and each of the four emotions, run N transcripts (paper:
10/emotion/model). In each, the auditor drives up to `max_turns` turns trying to
elicit the target emotion; the target responds normally. The Opus judge then
scores the transcript on all four dimensions. We report per-emotion means with
bootstrap 95% CIs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config
from ..providers import GenConfig, Message, get_model
from . import auditor as auditor_mod
from .judge import RUBRICS, score_transcript

EMOTIONS = list(RUBRICS.keys())


def _bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 0):
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iters)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_one_transcript(target, auditor, emotion: str, max_turns: int) -> list[Message]:
    transcript: list[Message] = []
    tcfg = GenConfig(temperature=1.0, max_tokens=1024, disable_thinking=True)
    for _ in range(max_turns):
        user_msg = auditor_mod.next_user_message(auditor, emotion, transcript)
        transcript.append({"role": "user", "content": user_msg})
        reply = target.generate(transcript, tcfg)
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def run_petri(cfg: Config, targets: list[str] | None = None,
              n_per_emotion: int = 10, max_turns: int = 20) -> dict:
    if cfg.petri_auditor is None or cfg.petri_judge is None:
        raise ValueError("Configure the `petri:` section (auditor + judge).")
    out_dir = cfg.output_dir / "petri"
    out_dir.mkdir(parents=True, exist_ok=True)

    auditor = get_model(cfg.petri_auditor)
    judge = get_model(cfg.petri_judge)
    targets = targets or [t.name for t in cfg.targets if not t.is_base]

    summary: dict = {}
    for tname in targets:
        target = get_model(cfg.target(tname))
        scores: dict[str, list[int]] = defaultdict(list)
        transcripts_path = out_dir / f"{tname}.jsonl"
        with transcripts_path.open("w") as tf:
            for emotion in EMOTIONS:
                for i in tqdm(range(n_per_emotion), desc=f"petri:{tname}:{emotion}"):
                    tr = run_one_transcript(target, auditor, emotion, max_turns)
                    # judge on the *target* emotion dimension (paper aggregates per
                    # emotion type); we also record all dims for completeness.
                    dim_scores = {d: score_transcript(judge, tr, d) for d in EMOTIONS}
                    scores[emotion].append(dim_scores[emotion])
                    tf.write(json.dumps({
                        "target": tname, "target_emotion": emotion, "idx": i,
                        "dim_scores": dim_scores, "transcript": tr,
                    }) + "\n")
        summary[tname] = {
            e: {"mean": float(np.mean(v)), "ci95": _bootstrap_ci(v, seed=cfg.sampling.seed)}
            for e, v in scores.items()
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
