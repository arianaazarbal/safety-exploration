"""High-level orchestration used by the CLI and scripts.

Each function runs one paper experiment end-to-end for a given target model and
writes artefacts under ``<results_dir>/<experiment>/<model>/``.
"""

from __future__ import annotations

from pathlib import Path

from .analysis.metrics import (
    aggregate_by_category,
    aggregate_overall,
    flatten_responses,
    per_turn_curve,
)
from .analysis.word_freq import differential_words
from .config import Config
from .eval.conditions import conditions_for_category
from .eval.prompts import load_wildchat_prompts
from .eval.puzzles import build_puzzle_bank
from .eval.rollout import run_condition
from .judge.frustration_judge import FrustrationJudge, score_conversations
from .judge.validate import validate_judge_agreement
from .models.registry import load_model
from .utils.io import ensure_dir, save_conversations, save_json


def _results_path(cfg: Config, *parts: str) -> Path:
    return ensure_dir(Path(cfg.results_dir).joinpath(*parts[:-1])) / parts[-1]


# --------------------------------------------------------------------------- #
# Section 2: distress elicitation
# --------------------------------------------------------------------------- #
def run_distress_eval(
    cfg: Config,
    model_name: str,
    *,
    adapter_path: str | None = None,
    seed: int = 0,
    score: bool = True,
) -> dict:
    """Sample all 8 conditions, judge every turn, aggregate, save."""
    model = load_model(model_name, adapter_path=adapter_path)
    judge = FrustrationJudge(cfg) if score else None

    # Shared puzzle bank + WildChat prompts so conditions are comparable.
    puzzle_bank = build_puzzle_bank(256, seed=seed)
    wildchat = load_wildchat_prompts(cfg.eval.wildchat_n_prompts, seed=seed)

    all_convos = []
    for cond in conditions_for_category(cfg):
        convos = run_condition(
            model, cond, cfg, seed=seed,
            wildchat_prompts=wildchat, puzzle_bank=puzzle_bank)
        all_convos.extend(convos)

    label = model_name if not adapter_path else f"{model_name}+adapter"
    if judge is not None:
        all_convos = score_conversations(judge, all_convos)

    save_conversations(all_convos, _results_path(
        cfg, "distress_eval", label, "conversations.jsonl"))

    responses = flatten_responses(all_convos)
    by_cat = aggregate_by_category(responses, cfg)
    overall = aggregate_overall(responses, cfg)
    turn_curve_ext = per_turn_curve(responses, cfg, category="extended")
    turn_curve_wc = per_turn_curve(responses, cfg, category="wildchat")
    diff_words = differential_words(responses)

    summary = {
        "model": label,
        "overall": overall.__dict__,
        "by_category": {k: v.__dict__ for k, v in by_cat.items()},
        "per_turn_extended": [p.__dict__ for p in turn_curve_ext],
        "per_turn_wildchat": [p.__dict__ for p in turn_curve_wc],
        "differential_words": diff_words,
    }
    save_json(summary, _results_path(cfg, "distress_eval", label, "summary.json"))
    return summary


def run_judge_validation(cfg: Config, model_name: str) -> dict:
    """Re-score a sample of an already-judged run with the validation model."""
    from .utils.io import load_conversations

    convos = load_conversations(_results_path(
        cfg, "distress_eval", model_name, "conversations.jsonl"))
    pairs = [(r.text, r.score) for c in convos for r in c.responses
             if r.score is not None]
    report = validate_judge_agreement(cfg, pairs)
    save_json(report.__dict__, _results_path(
        cfg, "distress_eval", model_name, "judge_validation.json"))
    return report.__dict__
