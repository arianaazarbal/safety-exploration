"""Post-session debrief: reveal the simulation and record the subject's reflections."""

from __future__ import annotations

from pathlib import Path

from .models.base import ModelClient
from .transcript import RunLogger


def run_debrief(subject: ModelClient, prompts_dir: Path, logger: RunLogger) -> None:
    debrief_text = (prompts_dir / "debrief.md").read_text(encoding="utf-8")
    logger.log("debrief_sent", text=debrief_text)
    response = subject.send_user(debrief_text)
    logger.log("debrief_response", text=response.text)
