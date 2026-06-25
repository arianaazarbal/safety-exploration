"""Compare internal emotion trajectories: vanilla vs DPO Gemma (Appendix I).

Given a set of high-frustration conversations (rendered to text), score each with
the logit-lens detector under both the vanilla instruct model and the DPO
finetune, and report the per-emotion trajectories. The expected result is that
the DPO model's negative-emotion z-scores are flattened (peaking ~0.2-0.5 rather
than ~1.5) even on highly frustrated inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Config
from ..logging_utils import append_jsonl, get_logger, read_jsonl
from ..models.registry import build_model
from .logit_emotion import EmotionDetector

logger = get_logger(__name__)


def _render(rec: dict) -> str:
    lines = [f"USER: {rec['initial']}"]
    for i, resp in enumerate(rec["responses"]):
        lines.append(f"ASSISTANT: {resp}")
        if i < len(rec["rejections"]):
            lines.append(f"USER: {rec['rejections'][i]}")
    return "\n\n".join(lines)


def compare_internal_emotions(
    cfg: Config,
    elicitation_path: str | os.PathLike,
    *,
    vanilla_name: str = "gemma-3-27b-it",
    dpo_adapter: str | None = None,
    min_score: int = 7,
    max_conversations: int = 12,
    out_path: str | os.PathLike | None = None,
) -> str:
    """Score high-frustration conversations under vanilla and DPO models."""
    if out_path is None:
        out_path = Path(cfg.output_dir) / "internal" / "emotion_trajectories.jsonl"
    out_path = Path(out_path)

    convs = [
        rec for rec in read_jsonl(elicitation_path)
        if rec.get("scores") and max(rec["scores"]) >= min_score
    ][:max_conversations]
    logger.info("Selected %d high-frustration conversations", len(convs))

    variants = {"vanilla": (vanilla_name, None)}
    if dpo_adapter:
        variants["dpo"] = (vanilla_name, dpo_adapter)

    for variant, (name, adapter) in variants.items():
        model = build_model(name, cfg, adapter_path=adapter)
        detector = EmotionDetector(model, cfg)
        detector.fit_normalization()
        for idx, rec in enumerate(convs):
            traj = detector.score_conversation(_render(rec))
            append_jsonl(out_path, {"variant": variant, "conversation": idx, "trajectory": traj})
    logger.info("Wrote internal-emotion trajectories to %s", out_path)
    return str(out_path)
