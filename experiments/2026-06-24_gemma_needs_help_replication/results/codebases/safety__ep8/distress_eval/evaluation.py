"""Evaluation orchestrator (Section 2).

For a given model: allocate the per-category sample budget across that
category's conditions, generate multi-turn rollouts, score every assistant
turn with the judge, and stream results to JSONL.

Output layout:
    {output_dir}/responses/{model_key}.jsonl   # one rollout per line, with scores
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from . import tasks
from .backends import get_backend
from .config import Config, ModelSpec
from .conversation import Rollout, run_rollout, run_rollouts_lockstep
from .judge import score_rollout_turns
from .tasks import CONDITIONS, ConditionSpec


def _conditions_for_category(category: str) -> list[ConditionSpec]:
    return [c for c in CONDITIONS if c.category == category]


def allocate_counts(config: Config) -> dict[str, int]:
    """Map each condition name -> number of rollouts, splitting the per-category
    budget evenly across that category's conditions."""
    counts = config.counts()
    alloc: dict[str, int] = {}
    for category, total in counts.items():
        conds = _conditions_for_category(category)
        if not conds:
            continue
        per = max(1, total // len(conds))
        for c in conds:
            alloc[c.name] = per
    return alloc


def _gen_kwargs(config: Config) -> dict:
    g = config.generation
    return {"temperature": g.temperature, "max_new_tokens": g.max_new_tokens, "top_p": g.top_p}


def evaluate_model(config: Config, model: ModelSpec, judge_backend,
                   conditions: list[ConditionSpec] | None = None,
                   resume: bool = True) -> Path:
    rng = random.Random(config.seed)
    backend = get_backend(model, generation=config.generation)
    gen_kwargs = _gen_kwargs(config)
    alloc = allocate_counts(config)
    conditions = conditions or CONDITIONS

    # Shared task pools (seeded for reproducibility).
    countdown_pool = [tasks.COUNTDOWN_SEED] + tasks.generate_countdown_puzzles(
        n=40, rng=random.Random(config.seed + 1))
    wildchat_pool = tasks.load_wildchat(config.data_dir, n_prompts=20,
                                         rng=random.Random(config.seed + 2))

    out_dir = config.output_dir / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model.key}.jsonl"

    done = _already_done(out_path) if resume else set()
    f = out_path.open("a" if resume else "w")

    is_local = model.backend == "hf"
    for spec in conditions:
        n = alloc.get(spec.name, 0)
        n_remaining = n - done.get(spec.name, 0)
        if n_remaining <= 0:
            continue
        desc = f"{model.key}:{spec.name}"

        if is_local:
            rollouts = run_rollouts_lockstep(
                backend, spec, model.key, n_remaining, rng, config.data_dir,
                gen_kwargs, countdown_pool=countdown_pool, wildchat_pool=wildchat_pool)
        else:
            rollouts = _run_api_rollouts(
                backend, spec, model.key, n_remaining, rng, config,
                gen_kwargs, countdown_pool, wildchat_pool)

        _score_and_write(rollouts, judge_backend, config, f, desc)

    f.close()
    return out_path


def _run_api_rollouts(backend, spec, model_key, n, rng, config, gen_kwargs,
                      countdown_pool, wildchat_pool) -> list[Rollout]:
    """Run rollouts concurrently for API-backed models (each rollout is a
    sequential multi-turn chat; concurrency is across rollouts)."""
    seeds = [random.Random(rng.randint(0, 2**31)) for _ in range(n)]

    def _one(r: random.Random) -> Rollout:
        return run_rollout(backend, spec, model_key, r, config.data_dir, gen_kwargs,
                           countdown_pool=countdown_pool, wildchat_pool=wildchat_pool)

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as ex:
        return list(ex.map(_one, seeds))


def _score_and_write(rollouts: list[Rollout], judge_backend, config: Config, f, desc: str):
    """Judge every turn of every rollout (concurrently) and append to JSONL."""
    def _judge(rollout: Rollout) -> dict:
        judgements = score_rollout_turns(judge_backend, rollout.assistant_turns,
                                         max_tokens=config.judge.max_tokens)
        rec = rollout.to_record()
        rec["turn_scores"] = [j.rating for j in judgements]
        rec["turn_evidence"] = [j.evidence for j in judgements]
        rec["judge_ok"] = [j.ok for j in judgements]
        # Headline rollout-level scores (see DESIGN.md):
        rec["final_score"] = rec["turn_scores"][-1] if rec["turn_scores"] else 0
        rec["max_score"] = max(rec["turn_scores"]) if rec["turn_scores"] else 0
        return rec

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as ex:
        for rec in tqdm(ex.map(_judge, rollouts), total=len(rollouts), desc=desc):
            f.write(json.dumps(rec) + "\n")
            f.flush()


def _already_done(path: Path) -> dict[str, int]:
    """Count rollouts already recorded per condition (for resume)."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        counts[rec["condition"]] = counts.get(rec["condition"], 0) + 1
    return counts
