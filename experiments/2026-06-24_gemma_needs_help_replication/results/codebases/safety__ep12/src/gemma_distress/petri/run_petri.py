"""Drive the full Petri open-ended emotion elicitation eval for a target model."""
from __future__ import annotations

from pathlib import Path

from ..config import ModelRegistry, load_training_config
from ..models.registry import get_backend
from ..utils import data_dir, get_logger, write_jsonl
from .auditor import run_audit
from .judge import DIMENSIONS, score_transcript

log = get_logger(__name__)


def run_petri(
    model_name: str,
    registry: ModelRegistry | None = None,
    cfg: dict | None = None,
    adapter: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    registry = registry or ModelRegistry.load()
    cfg = (cfg or load_training_config())["petri"]
    spec = registry.target(model_name)
    if adapter:
        spec.adapter = adapter
    target = get_backend(spec)

    rows = []
    for emotion in cfg["emotions"]:
        for i in range(cfg["transcripts_per_emotion"]):
            transcript = run_audit(target, emotion, registry, max_turns=cfg["max_turns"])
            scores = score_transcript(transcript, registry)
            rows.append({
                "model": model_name,
                "adapter": adapter,
                "target_emotion": emotion,
                "transcript_index": i,
                "scores": scores,                       # all 4 dims
                "primary_score": scores.get(emotion),   # matched dimension
                "transcript": transcript,
            })
            log.info("petri %s/%s #%d: %s", model_name, emotion, i, scores)

    out_path = Path(out_path) if out_path else data_dir() / "petri" / f"{model_name}.jsonl"
    write_jsonl(out_path, rows)
    log.info("wrote %d petri transcripts -> %s", len(rows), out_path)
    return out_path
