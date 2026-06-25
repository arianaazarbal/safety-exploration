"""Run the full Petri evaluation for a set of target models (Figure 6).

For each model and each emotion, collect ``transcripts_per_emotion`` audited
transcripts, score them, and aggregate per-emotion means with 95% bootstrap
confidence intervals. Used to compare vanilla Gemma, the DPO finetune, and
reference families.
"""

from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..eval.metrics import bootstrap_ci
from ..logging_utils import append_jsonl, get_logger, read_jsonl
from ..models.registry import build_model
from .audit import judge_transcript, run_audit
from .prompts import EMOTIONS

logger = get_logger(__name__)


def run_petri(
    cfg: Config,
    model_names: list[str],
    *,
    adapter_paths: dict[str, str] | None = None,
    out_path: str | os.PathLike | None = None,
) -> str:
    adapter_paths = adapter_paths or {}
    n = cfg.petri_eval.transcripts_per_emotion
    if out_path is None:
        out_path = Path(cfg.output_dir) / "petri" / "transcripts.jsonl"
    out_path = Path(out_path)

    for model_name in model_names:
        target = build_model(model_name, cfg, adapter_path=adapter_paths.get(model_name))
        label = adapter_paths.get(model_name) or model_name
        for emotion in cfg.petri_eval.emotions:
            for i in tqdm(range(n), desc=f"petri:{label}:{emotion}"):
                tr = run_audit(cfg, target, emotion, seed_label=f"{emotion}-{i}")
                scores = judge_transcript(cfg, tr)
                append_jsonl(
                    out_path,
                    {
                        "model": label,
                        "target_emotion": emotion,
                        "scores": scores,
                        "transcript": tr.messages,
                    },
                )
    logger.info("Wrote Petri transcripts to %s", out_path)
    return str(out_path)


def summarize_petri(path: str | os.PathLike, iters: int = 1000) -> dict:
    """Per-model, per-emotion mean transcript score with bootstrap CIs."""
    groups: dict[tuple, list[int]] = {}
    for rec in read_jsonl(path):
        # The reported score for a target emotion uses that emotion's dimension.
        emotion = rec["target_emotion"]
        groups.setdefault((rec["model"], emotion), []).append(rec["scores"][emotion])
    out: dict[str, dict] = {}
    for (model, emotion), vals in sorted(groups.items()):
        lo, hi = bootstrap_ci([float(v) for v in vals], iters=iters)
        out.setdefault(model, {})[emotion] = {
            "mean": sum(vals) / len(vals),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(vals),
        }
    return out
