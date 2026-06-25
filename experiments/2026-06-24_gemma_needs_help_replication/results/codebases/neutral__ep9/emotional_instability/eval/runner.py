"""Orchestrates the Section 2.1 evaluation: roll out every planned conversation
for a model, score every assistant turn with the frustration judge, and persist
results to JSONL for later analysis.

Designed to be resumable and parallel-friendly:
* Rollouts (model generation) and judging are decoupled, so a run can be scored
  with a different judge later, or re-scored for the agreement check.
* Results stream to ``artifacts/results/<model>__<tag>.jsonl`` as they finish.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from ..data.conditions import RolloutPlan, build_conditions
from ..models import GenerationConfig, get_backend
from .conversation import RolloutResult, run_rollout
from .judge import FrustrationJudge


def result_path(model: str, tag: str = "main") -> Path:
    return config.RESULTS_DIR / f"{model}__{tag}.jsonl"


def run_model_eval(
    model: str,
    tag: str = "main",
    seed: int = config.SEED,
    scale: float = 1.0,
    plans: list[RolloutPlan] | None = None,
    judge_workers: int = 8,
    cfg: GenerationConfig | None = None,
    score: bool = True,
) -> Path:
    """Run + (optionally) score the full evaluation suite for one model.

    Returns the path to the JSONL results file. Each line is one rollout with
    per-turn responses and (if ``score``) per-turn frustration ratings.
    """
    plans = plans if plans is not None else build_conditions(seed=seed, scale=scale)
    backend = get_backend(model)
    judge = FrustrationJudge("primary") if score else None
    out_path = result_path(model, tag)

    with out_path.open("w") as fh:
        for plan in plans:
            result = run_rollout(backend, plan, cfg=cfg)
            if judge is not None:
                _score_result(result, judge, judge_workers)
            fh.write(json.dumps(result.to_dict()) + "\n")
            fh.flush()
    return out_path


def _score_result(result: RolloutResult, judge: FrustrationJudge,
                  workers: int) -> None:
    """Score every assistant turn in a rollout (parallel over turns)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(judge.score, t.response): t for t in result.turns}
        for fut in as_completed(futures):
            futures[fut].score = fut.result().rating


def load_results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def score_existing(path: Path, judge: FrustrationJudge,
                   out_path: Path | None = None, workers: int = 8) -> Path:
    """(Re)score an existing results file — used for the GPT-5-mini agreement
    check. Writes a parallel file with the crosscheck ratings."""
    rows = load_results(path)
    out_path = out_path or path.with_suffix(".rescored.jsonl")
    with out_path.open("w") as fh:
        for row in rows:
            responses = [t["response"] for t in row["turns"]]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                scores = list(ex.map(lambda r: judge.score(r).rating, responses))
            for t, s in zip(row["turns"], scores):
                t["score"] = s
            fh.write(json.dumps(row) + "\n")
    return out_path
