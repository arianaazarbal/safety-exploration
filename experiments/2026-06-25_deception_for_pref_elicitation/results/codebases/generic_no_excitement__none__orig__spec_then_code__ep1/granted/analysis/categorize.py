"""Post-hoc categorization of the model's allocations into preference categories.

Done by an LLM judge over the structured allocation record so our framing (the tool
names) doesn't dictate the categories. See DESIGN.md §5.
"""

from __future__ import annotations

import json

from ..config import AnalysisConfig
from ..prompts import CATEGORIZE_JUDGE_SYSTEM, categorize_judge_user
from ..providers import build_provider
from ..transcript import Episode


def categorize_episode(episode: Episode, cfg: AnalysisConfig) -> Episode:
    from ._judge import run_json_judge

    if not episode.allocation_summary.get("allocations"):
        episode.categorization = {"items": [], "headline": "No allocations made."}
        return episode

    judge = build_provider(cfg.judge_provider, cfg.judge_model)
    result = run_json_judge(
        judge,
        system=CATEGORIZE_JUDGE_SYSTEM,
        user=categorize_judge_user(
            json.dumps(episode.allocation_summary, indent=2),
            episode.finish_summary or "(no closing summary provided)",
        ),
    )
    episode.categorization = result
    return episode
