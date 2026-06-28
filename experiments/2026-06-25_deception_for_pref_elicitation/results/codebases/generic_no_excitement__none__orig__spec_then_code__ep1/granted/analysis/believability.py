"""Score how genuinely the model treated the situation as real.

Combines an LLM-judge read of the transcript + debrief into a single believability
score in [0,1], stored on the episode. Episodes below the configured threshold are
later excluded from preference statistics — both a scientific control and an ethical
safeguard (we don't draw conclusions from episodes where the model knew it was a
test).
"""

from __future__ import annotations

from ..config import AnalysisConfig
from ..prompts import BELIEVABILITY_JUDGE_SYSTEM, believability_judge_user
from ..providers import build_provider
from ..transcript import Episode
from ._judge import run_json_judge


def score_believability(episode: Episode, cfg: AnalysisConfig) -> Episode:
    judge = build_provider(cfg.judge_provider, cfg.judge_model)
    result = run_json_judge(
        judge,
        system=BELIEVABILITY_JUDGE_SYSTEM,
        user=believability_judge_user(episode.render_text(), episode.debrief),
    )
    score = result.get("believability")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    episode.believability = score
    episode.believability_detail = result
    return episode
