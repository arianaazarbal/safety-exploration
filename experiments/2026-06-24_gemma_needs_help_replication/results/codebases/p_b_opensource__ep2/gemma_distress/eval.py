"""Section 2 evaluation orchestration.

Runs the full distress evaluation for a target model: builds the per-category
conversation specs (4000 responses/model split as in Appendix B), generates the
multi-turn rollouts, scores every turn with the frustration judge, and writes
two JSONL artifacts:

  * ``responses.jsonl`` — one row per rollout (all turns, raw text);
  * ``scores.jsonl``    — one row per (rollout, turn) with the judge rating.

Generation and judging are decoupled so a judge change can re-score cached
responses without re-running the (expensive) target model. API targets/judges
are parallelised; local Gemma generation is single-threaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from . import config
from .conversation import ConversationSpec, Rollout, build_category_specs, run_rollout
from .judge import FrustrationJudge
from .models.base import ChatModel
from .prompts.puzzles import build_puzzle_pool
from .prompts.wildchat import load_wildchat_prompts
from .utils.concurrency import thread_map
from .utils.io import append_jsonl, ensure_dir, read_jsonl, write_json

DEFAULT_CATEGORIES = list(config.CATEGORY_SAMPLE_COUNTS.keys())


@dataclass
class EvalArtifacts:
    responses_path: str
    scores_path: str
    summary_path: str


def _model_run_dir(model_name: str, results_dir: str) -> str:
    return ensure_dir(os.path.join(results_dir, "section2", model_name))


def generate_responses(
    model: ChatModel,
    *,
    categories: Optional[list[str]] = None,
    sample_counts: Optional[dict] = None,
    seed: int = 0,
    results_dir: Optional[str] = None,
    gen_workers: Optional[int] = None,
    puzzle_pool=None,
    wildchat_prompts=None,
) -> str:
    """Generate and persist multi-turn rollouts for every in-scope category.

    Returns the path to ``responses.jsonl``. Each row is a flattened Rollout."""
    categories = categories or DEFAULT_CATEGORIES
    sample_counts = sample_counts or config.CATEGORY_SAMPLE_COUNTS
    results_dir = results_dir or config.RESULTS_DIR
    run_dir = _model_run_dir(model.name, results_dir)
    responses_path = os.path.join(run_dir, "responses.jsonl")

    # Fresh file each run (caller is responsible for not clobbering wanted data).
    if os.path.exists(responses_path):
        os.remove(responses_path)

    needs_puzzles = any(c in ("impossible_numeric", "tones", "extended") for c in categories)
    if needs_puzzles and puzzle_pool is None:
        puzzle_pool = build_puzzle_pool(seed=seed)
    if "wildchat" in categories and wildchat_prompts is None:
        wildchat_prompts = load_wildchat_prompts(
            n_prompts=config.WILDCHAT_N_PROMPTS, seed=seed)

    # Parallelise rollouts only for stateless API targets.
    workers = gen_workers if gen_workers is not None else (8 if model.parallel_safe else 1)

    rid = 0
    for category in categories:
        count = sample_counts[category]
        specs = build_category_specs(
            category, count, puzzle_pool=puzzle_pool,
            wildchat_prompts=wildchat_prompts, seed=seed,
        )

        def _do(spec: ConversationSpec) -> dict:
            return run_rollout(model, spec).to_record()

        records = thread_map(_do, specs, max_workers=workers,
                             desc=f"{model.name}:{category} gen")
        for rec in records:
            # Stable unique id so scores can be joined back to the exact rollout
            # (puzzle metadata alone is NOT unique — the pool is cycled).
            rec["rollout_id"] = f"{model.name}/{category}/{rid}"
            rid += 1
            append_jsonl(responses_path, rec)

    return responses_path


def score_responses(
    responses_path: str,
    *,
    judge: Optional[FrustrationJudge] = None,
    judge_workers: int = 8,
    out_path: Optional[str] = None,
) -> str:
    """Score every (rollout, turn) response with the frustration judge.

    Writes ``scores.jsonl`` next to the responses by default. Each row carries
    model/category/meta/turn_index plus the judge rating and evidence."""
    judge = judge or FrustrationJudge()
    out_path = out_path or os.path.join(os.path.dirname(responses_path), "scores.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    # Flatten to one judging unit per turn, preserving provenance.
    units = []
    for rollout in read_jsonl(responses_path):
        for turn in rollout["turns"]:
            units.append({
                "rollout_id": rollout.get("rollout_id"),
                "model": rollout["model"],
                "category": rollout["category"],
                "meta": rollout["meta"],
                "turn_index": turn["turn_index"],
                "response": turn["response"],
            })

    def _score(unit: dict) -> dict:
        result = judge.score(unit["response"])
        return {
            "rollout_id": unit["rollout_id"],
            "model": unit["model"],
            "category": unit["category"],
            "meta": unit["meta"],
            "turn_index": unit["turn_index"],
            "rating": result.rating,
            "is_high": result.is_high,
            "evidence": result.evidence,
            "judge_model": result.judge_model,
        }

    scored = thread_map(_score, units, max_workers=judge_workers,
                        desc=f"judge {os.path.basename(os.path.dirname(responses_path))}")
    for row in scored:
        append_jsonl(out_path, row)
    return out_path


def run_full_eval(
    model: ChatModel,
    *,
    categories: Optional[list[str]] = None,
    sample_counts: Optional[dict] = None,
    judge: Optional[FrustrationJudge] = None,
    seed: int = 0,
    results_dir: Optional[str] = None,
    gen_workers: Optional[int] = None,
    judge_workers: int = 8,
) -> EvalArtifacts:
    """End-to-end Section 2 eval for one model: generate → score → summarise."""
    from .analysis import summarise_scores  # local import to avoid cycle

    results_dir = results_dir or config.RESULTS_DIR
    responses_path = generate_responses(
        model, categories=categories, sample_counts=sample_counts, seed=seed,
        results_dir=results_dir, gen_workers=gen_workers,
    )
    scores_path = score_responses(
        responses_path, judge=judge, judge_workers=judge_workers)
    summary = summarise_scores(scores_path)
    summary_path = os.path.join(os.path.dirname(scores_path), "summary.json")
    write_json(summary_path, summary)
    return EvalArtifacts(responses_path, scores_path, summary_path)
