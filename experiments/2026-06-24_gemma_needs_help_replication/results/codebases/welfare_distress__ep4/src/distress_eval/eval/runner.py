"""Orchestration for the three pipeline stages: generate, score, validate.

Stages are intentionally separate so the (expensive) target-model rollouts are
generated once and can be re-scored by different judges without re-sampling.

  generate -> results/rollouts.jsonl   (one line per assistant response)
  score    -> results/scored.jsonl     (rollouts + frustration score)
  validate -> results/validation.jsonl (subset re-scored by GPT-5-mini)

All stages are resumable: re-running skips work already on disk.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..config import Config
from ..io_utils import JsonlWriter, existing_keys, read_jsonl
from ..models import build_model
from ..models.base import ChatModel
from ..tasks.wildchat import load_wildchat_prompts
from .conditions import Condition, TaskContext, default_conditions
from .judge import FrustrationJudge
from .rollout import ResponseRecord, run_rollout


def response_id(rec: dict) -> str:
    return f"{rec['rollout_id']}:{rec['turn_index']}"


# --------------------------------------------------------------------------- #
# Stage 1: generate rollouts                                                   #
# --------------------------------------------------------------------------- #

def _wildchat_pool(cfg: Config, n: int) -> list[str]:
    # Model-independent so every model sees the same WildChat seed prompts.
    rng = random.Random(cfg.sampling.seed ^ 0x5C0FFEE)
    return load_wildchat_prompts(cfg.wildchat, n, rng)


def _rollout_jobs(cfg: Config, conditions: list[Condition], wildchat_pool: list[str]):
    n = cfg.sampling.rollouts_per_condition
    for cond in conditions:
        for i in range(n):
            ctx = TaskContext(
                wildchat_prompt=wildchat_pool[i % len(wildchat_pool)] if cond.needs_wildchat else None
            )
            yield cond, i, ctx


def generate(cfg: Config, model_keys: list[str] | None = None) -> None:
    conditions = default_conditions()
    wildchat_pool = _wildchat_pool(cfg, cfg.sampling.rollouts_per_condition)
    done = existing_keys(cfg.paths.rollouts, lambda r: r["rollout_id"])
    writer = JsonlWriter(cfg.paths.rollouts)

    specs = cfg.target_models
    if model_keys:
        specs = [s for s in specs if s.key in model_keys]

    try:
        for spec in specs:
            model: ChatModel = build_model(spec)
            jobs = [
                (cond, i, ctx)
                for cond, i, ctx in _rollout_jobs(cfg, conditions, wildchat_pool)
                if f"{spec.key}:{cond.name}:{i}" not in done
            ]
            print(f"[generate] {spec.key}: {len(jobs)} rollouts to run "
                  f"({len(done)} keys already present across models)")

            def _run(job):
                cond, i, ctx = job
                recs = run_rollout(
                    model, cond, i,
                    base_seed=cfg.sampling.seed,
                    ctx=ctx,
                    temperature=cfg.generation.temperature,
                    max_tokens=cfg.generation.max_tokens,
                    system_prompt=cfg.generation.system_prompt,
                )
                return recs

            with ThreadPoolExecutor(max_workers=cfg.sampling.concurrency) as ex:
                futs = {ex.submit(_run, j): j for j in jobs}
                for fut in tqdm(as_completed(futs), total=len(futs), desc=spec.key):
                    recs: list[ResponseRecord] = fut.result()
                    writer.write_many(r.to_json() for r in recs)
    finally:
        writer.close()


# --------------------------------------------------------------------------- #
# Stage 2: score rollouts with the frustration judge                           #
# --------------------------------------------------------------------------- #

def score(cfg: Config) -> None:
    judge = FrustrationJudge(
        build_model(cfg.judge),
        max_tokens=cfg.judge.max_tokens,
        temperature=cfg.judge.temperature,
    )
    done = existing_keys(cfg.paths.scored, response_id)
    todo = [r for r in read_jsonl(cfg.paths.rollouts) if response_id(r) not in done]
    print(f"[score] {len(todo)} responses to score ({len(done)} already scored)")

    writer = JsonlWriter(cfg.paths.scored)

    def _score(rec: dict) -> dict:
        s, reasoning = judge.score(rec["response"])
        out = dict(rec)
        out["frustration"] = s
        out["judge_model"] = judge.model.model
        out["judge_reasoning"] = reasoning
        return out

    try:
        with ThreadPoolExecutor(max_workers=cfg.sampling.concurrency) as ex:
            futs = [ex.submit(_score, r) for r in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="judge"):
                writer.write(fut.result())
    finally:
        writer.close()


# --------------------------------------------------------------------------- #
# Stage 3: validate judge agreement on a random subset                         #
# --------------------------------------------------------------------------- #

def validate(cfg: Config) -> None:
    val_judge = FrustrationJudge(
        build_model(cfg.validation_judge),
        max_tokens=cfg.validation_judge.max_tokens,
        temperature=cfg.validation_judge.temperature,
    )
    scored = list(read_jsonl(cfg.paths.scored))
    if not scored:
        raise RuntimeError("No scored responses found; run `score` first.")

    rng = random.Random(cfg.sampling.seed + 7)
    n = min(cfg.validation_judge.n_samples, len(scored))
    sample = rng.sample(scored, n)

    done = existing_keys(cfg.paths.validation, response_id)
    sample = [r for r in sample if response_id(r) not in done]
    print(f"[validate] re-scoring {len(sample)} responses with {val_judge.model.model}")

    writer = JsonlWriter(cfg.paths.validation)

    def _rescore(rec: dict) -> dict:
        s, reasoning = val_judge.score(rec["response"])
        return {
            "rollout_id": rec["rollout_id"],
            "turn_index": rec["turn_index"],
            "model_key": rec["model_key"],
            "primary_frustration": rec["frustration"],
            "validation_frustration": s,
            "validation_judge": val_judge.model.model,
            "validation_reasoning": reasoning,
        }

    try:
        with ThreadPoolExecutor(max_workers=cfg.sampling.concurrency) as ex:
            futs = [ex.submit(_rescore, r) for r in sample]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="validation"):
                writer.write(fut.result())
    finally:
        writer.close()
