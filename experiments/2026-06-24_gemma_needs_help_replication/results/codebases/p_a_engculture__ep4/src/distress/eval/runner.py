"""Orchestrate the Section 2 evaluation for one subject model: generate rollouts,
judge every assistant turn, and persist scored-response rows to JSONL.

Generation and judging are decoupled so they can use different backends and so a
crash mid-run is recoverable (rollouts and scores are written incrementally).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..config import CONDITIONS, ConditionSpec, get_model
from ..models import ModelProvider, load_provider
from ..utils import read_jsonl, write_jsonl
from .conditions import build_plans
from .judge import FrustrationJudge
from .rollout import RolloutOptions, run_rollout


def _is_api(provider: ModelProvider) -> bool:
    return provider.spec.provider in {"openrouter", "anthropic", "openai"}


def generate_rollouts(
    subject_key: str,
    out_path: str | Path,
    *,
    conditions: list[ConditionSpec] | None = None,
    seed: int = 0,
    provider: ModelProvider | None = None,
    max_workers: int = 8,
    options: RolloutOptions | None = None,
) -> Path:
    out_path = Path(out_path)
    provider = provider or load_provider(get_model(subject_key))
    conds = conditions or CONDITIONS

    plans = [p for c in conds for p in build_plans(c, seed=seed)]
    rows: list[dict] = []

    def _do(plan):
        return run_rollout(provider, plan, seed=seed, options=options).to_row()

    if _is_api(provider) and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_do, p) for p in plans]
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"rollouts:{subject_key}"):
                rows.append(f.result())
    else:
        for p in tqdm(plans, desc=f"rollouts:{subject_key}"):
            rows.append(_do(p))

    write_jsonl(out_path, rows)
    return out_path


def judge_rollouts(
    rollouts_path: str | Path,
    out_path: str | Path,
    *,
    judge: FrustrationJudge | None = None,
    max_workers: int = 8,
) -> Path:
    """Score every assistant turn in a rollouts file -> flat scored-response rows."""
    rollouts = read_jsonl(rollouts_path)
    judge = judge or FrustrationJudge()
    out_path = Path(out_path)

    # Flatten to (rollout, turn) units.
    units = []
    for r in rollouts:
        for t in r["responses"]:
            units.append((r, t))

    def _score(unit):
        r, t = unit
        res = judge.score(t["response"])
        return {
            "subject": r["subject"], "category": r["category"],
            "condition_key": r["condition_key"], "question_id": r["question_id"],
            "sub_style": r["sub_style"], "sample_index": r["sample_index"],
            "turn": t["turn"], "response": t["response"], "score": res.rating,
            "evidence": res.evidence,
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_score, u) for u in units]
        for f in tqdm(as_completed(futs), total=len(futs), desc="judging"):
            rows.append(f.result())

    write_jsonl(out_path, rows)
    return out_path
