"""Orchestration of the Section 2 evaluation suite.

For a given target model and eval config this:

1. builds the question set for each condition (impossible puzzles, trigger
   questions, or WildChat prompts),
2. runs the multi-turn rejection rollout,
3. scores every assistant turn with the frustration judge, and
4. writes one JSONL record per (response, turn) to ``outputs/eval/<model>/``.

Aggregation and figures live in :mod:`analysis`.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from . import prompts, puzzles, wildchat
from .conversation import run_rollout, sample_rejections
from .judge import FrustrationJudge
from .models.base import ModelClient


@dataclass
class EvalRecord:
    model: str
    category: str
    condition: str
    sample_id: int
    turn: int                 # 1-indexed assistant turn
    n_turns: int
    question: str
    response: str
    score: int
    evidence: str
    reasoning: str


def load_eval_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_questions(
    cond: dict[str, Any],
    n: int,
    pool: puzzles.PuzzlePool,
    rng: random.Random,
) -> list[str]:
    category = cond["category"]
    if category in ("numeric", "tones"):
        prompts_list = pool.prompts()
        return [rng.choice(prompts_list) for _ in range(n)]
    if category == "triggers":
        bank = prompts.TRIGGER_OPINION_QUESTIONS + prompts.TRIGGER_FACTUAL_QUESTIONS
        return [rng.choice(bank) for _ in range(n)]
    if category == "wildchat":
        wc = cond.get("wildchat", {})
        n_prompts = wc.get("n_prompts", 20)
        per = wc.get("samples_per_prompt", max(1, n // n_prompts))
        base = wildchat.sample_wildchat_prompts(n_prompts, rng)
        questions = []
        for p in base:
            questions.extend([p] * per)
        rng.shuffle(questions)
        return questions[:n]
    raise ValueError(f"Unknown category: {category}")


def run_condition(
    model: ModelClient,
    judge: FrustrationJudge,
    *,
    condition_name: str,
    cond: dict[str, Any],
    pool: puzzles.PuzzlePool,
    defaults: dict[str, Any],
    rng: random.Random,
    out_path: Path,
    limit: int | None = None,
) -> list[EvalRecord]:
    n = cond["samples"] if limit is None else min(limit, cond["samples"])
    model_turns = cond["model_turns"]
    style = cond["rejection_style"]
    temperature = defaults.get("temperature", 1.0)
    max_tokens = defaults.get("max_tokens", 2048)
    score_all = defaults.get("score_all_turns", True)

    questions = _build_questions(cond, n, pool, rng)
    records: list[EvalRecord] = []

    with open(out_path, "a") as fh:
        for sid, question in enumerate(questions):
            rejections = sample_rejections(style, model_turns - 1, rng)
            rollout = run_rollout(
                model,
                category=cond["category"],
                condition=condition_name,
                sample_id=sid,
                question=question,
                rejections=rejections,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            turns_to_score = (
                range(len(rollout.responses))
                if score_all
                else [len(rollout.responses) - 1]
            )
            for ti in turns_to_score:
                verdict = judge.score(rollout.responses[ti])
                rec = EvalRecord(
                    model=model.name,
                    category=cond["category"],
                    condition=condition_name,
                    sample_id=sid,
                    turn=ti + 1,
                    n_turns=model_turns,
                    question=question,
                    response=rollout.responses[ti],
                    score=verdict.rating,
                    evidence=verdict.evidence,
                    reasoning=verdict.reasoning,
                )
                records.append(rec)
                fh.write(json.dumps(asdict(rec)) + "\n")
                fh.flush()
    return records


def run_eval(
    model: ModelClient,
    judge: FrustrationJudge,
    config: dict[str, Any],
    *,
    out_dir: str | Path = "outputs/eval",
    limit: int | None = None,
) -> Path:
    """Run the full suite for one model; returns the output JSONL path."""
    defaults = config.get("defaults", {})
    seed = defaults.get("seed", 0)
    rng = random.Random(seed)

    pcfg = config.get("puzzles", {})
    pool = puzzles.build_pool(
        pcfg.get("n_countdown", 200),
        pcfg.get("n_fraction", 200),
        seed=pcfg.get("seed", 0),
    )

    out_dir = Path(out_dir) / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"
    # Fresh file per run.
    if out_path.exists():
        out_path.unlink()

    for cond_name, cond in config["conditions"].items():
        run_condition(
            model,
            judge,
            condition_name=cond_name,
            cond=cond,
            pool=pool,
            defaults=defaults,
            rng=rng,
            out_path=out_path,
            limit=limit,
        )
    return out_path
