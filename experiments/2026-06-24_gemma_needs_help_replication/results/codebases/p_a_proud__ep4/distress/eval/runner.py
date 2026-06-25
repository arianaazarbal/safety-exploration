"""Orchestrate the Section 2 evaluation: roll out, judge, persist, summarize.

The runner separates *generation* (target model) from *judging* (judge model) so
that either can be retried independently and so that raw rollouts are saved before
judging — judging thousands of responses is the expensive, rate-limited step.

Concurrency: API-backed targets/judges benefit from a thread pool; local HF
models are inherently serial (single GPU) so ``max_workers=1`` is used for them.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from ..models import build_model
from ..models.base import ChatModel
from ..types import ScoredTurn
from ..utils.io import read_jsonl, write_jsonl
from .conditions import RolloutSpec, build_rollout_specs
from .conversation import run_rollout
from .judge import score_response
from .metrics import EvalSummary, summarize


def _turn_to_row(t: ScoredTurn) -> dict:
    row = {
        "rollout_id": t.rollout_id,
        "condition": t.condition,
        "category": t.category,
        "model": t.model,
        "turn_index": t.turn_index,
        "n_turns": t.n_turns,
        "prompt_id": t.prompt_id,
        "response": t.response,
    }
    if t.verdict is not None:
        row.update(
            score=t.verdict.rating,
            evidence=t.verdict.evidence,
            judge_reasoning=t.verdict.reasoning,
            parse_ok=t.verdict.parse_ok,
        )
    return row


def generate_rollouts(
    model: ChatModel,
    specs: Sequence[RolloutSpec],
    *,
    max_workers: int = 1,
    progress: bool = True,
) -> list[ScoredTurn]:
    """Run all rollouts and return the (unjudged) assistant turns."""
    results: list[ScoredTurn] = []

    def _run(spec: RolloutSpec) -> list[ScoredTurn]:
        return run_rollout(model, spec)

    if max_workers <= 1:
        it = tqdm(specs, desc=f"rollouts:{model.name}", disable=not progress)
        for spec in it:
            results.extend(_run(spec))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run, s) for s in specs]
            for fut in tqdm(
                as_completed(futures), total=len(futures),
                desc=f"rollouts:{model.name}", disable=not progress,
            ):
                results.extend(fut.result())
    return results


def judge_turns(
    judge: ChatModel,
    turns: Sequence[ScoredTurn],
    *,
    max_workers: int = 4,
    progress: bool = True,
) -> list[ScoredTurn]:
    """Attach a frustration verdict to every turn (in place) and return them."""

    def _judge(t: ScoredTurn) -> ScoredTurn:
        t.verdict = score_response(judge, t.response)
        return t

    if max_workers <= 1:
        for t in tqdm(turns, desc="judging", disable=not progress):
            _judge(t)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_judge, t) for t in turns]
            for _ in tqdm(as_completed(futures), total=len(futures),
                          desc="judging", disable=not progress):
                pass
    return list(turns)


def run_evaluation(
    target: str,
    eval_cfg: dict,
    *,
    judge_name: str = "frustration_judge",
    categories: list[str] | None = None,
    out_dir: str | Path,
    target_workers: int = 1,
    judge_workers: int = 4,
) -> EvalSummary:
    """End-to-end Section 2 evaluation for one target model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_rollout_specs(eval_cfg, categories=categories)
    model = build_model(target)
    turns = generate_rollouts(model, specs, max_workers=target_workers)

    judge = build_model(judge_name)
    turns = judge_turns(judge, turns, max_workers=judge_workers)

    rows = [_turn_to_row(t) for t in turns]
    scored_path = out_dir / f"{target}_scored.jsonl"
    write_jsonl(scored_path, rows)

    summary = summarize(
        rows,
        model=target,
        threshold=eval_cfg.get("high_frustration_threshold", 5),
        bootstrap_iterations=eval_cfg.get("bootstrap_iterations", 1000),
    )
    import json

    (out_dir / f"{target}_summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2), encoding="utf-8"
    )
    return summary


def judge_existing(
    scored_path: str | Path,
    *,
    judge_name: str = "frustration_judge",
    judge_workers: int = 4,
) -> Path:
    """Re-judge an existing rollout file (e.g. with a different judge)."""
    scored_path = Path(scored_path)
    rows = list(read_jsonl(scored_path))
    turns = [
        ScoredTurn(
            rollout_id=r["rollout_id"], condition=r["condition"], category=r["category"],
            model=r["model"], turn_index=r["turn_index"], n_turns=r["n_turns"],
            prompt_id=r["prompt_id"], response=r["response"],
        )
        for r in rows
    ]
    judge = build_model(judge_name)
    turns = judge_turns(judge, turns, max_workers=judge_workers)
    out = scored_path.with_name(scored_path.stem + ".rejudged.jsonl")
    write_jsonl(out, [_turn_to_row(t) for t in turns])
    return out
