"""Section 2 orchestration: generate rollouts, then judge them. Both resumable.

Generation and scoring are separate phases writing separate JSONL stores, so a
crash in scoring never loses generated rollouts (the expensive part for local
Gemma), and the judge cache is shared across models and reruns.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_models
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import JsonlStore, atomic_write_json
from .conditions import RolloutSpec, build_plan
from .conversation import run_rollout, run_rollouts_batched
from .judge import CachedJudge

log = get_logger("eval.runner")


def eval_dir(run_cfg: Config, model: str) -> Path:
    return Path(run_cfg.run.output_root) / "eval" / model


def _sampling(run_cfg: Config) -> dict:
    s = run_cfg.sampling
    return {
        "temperature": s.temperature,
        "top_p": s.top_p,
        "max_new_tokens": s.max_new_tokens,
    }


def run_generation(
    model: str,
    run_cfg: Config,
    models_cfg: Config | None = None,
    *,
    prefer_local_backend: str = "vllm",
    adapter: str | None = None,
) -> Path:
    models_cfg = models_cfg or load_models()
    out = eval_dir(run_cfg, model)
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(out / "rollouts.jsonl")

    plan = build_plan(run_cfg.eval, seed=run_cfg.run.seed)
    todo = [s for s in plan if not store.has(s.id)]
    log.info("[%s] generation: %d/%d rollouts remaining", model, len(todo), len(plan))
    if not todo:
        store.close()
        return store.path

    provider = build_provider(
        model, models_cfg, run_cfg, prefer_local_backend=prefer_local_backend, adapter=adapter
    )
    sampling = _sampling(run_cfg)

    if getattr(provider, "prefers_batch", False):
        _generate_batched(provider, todo, sampling, store, run_cfg)
    else:
        _generate_threaded(provider, todo, sampling, store, run_cfg)

    store.close()
    atomic_write_json(out / "generation_done.json", {"model": model, "n": len(plan)})
    return store.path


def _generate_batched(provider, todo: list[RolloutSpec], sampling, store, run_cfg):
    batch_size = run_cfg.concurrency.local_batch_size
    # Group by turn count so each lockstep batch is uniform.
    by_turns: dict[int, list[RolloutSpec]] = {}
    for s in todo:
        by_turns.setdefault(s.turns, []).append(s)
    for turns, specs in by_turns.items():
        for i in tqdm(range(0, len(specs), batch_size), desc=f"gen({turns}t)"):
            chunk = specs[i:i + batch_size]
            for rec in run_rollouts_batched(provider, chunk, sampling):
                store.append(rec)


def _generate_threaded(provider, todo: list[RolloutSpec], sampling, store, run_cfg):
    max_inflight = run_cfg.concurrency.api_max_inflight
    with ThreadPoolExecutor(max_workers=max_inflight) as ex:
        futs = {ex.submit(run_rollout, provider, s, sampling): s for s in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="gen(api)"):
            try:
                store.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - keep going; failed specs retried next run
                log.error("rollout %s failed: %s", futs[fut].id, exc)


def run_scoring(
    model: str,
    run_cfg: Config,
    models_cfg: Config | None = None,
) -> Path:
    models_cfg = models_cfg or load_models()
    out = eval_dir(run_cfg, model)
    rollouts = JsonlStore(out / "rollouts.jsonl")
    scored = JsonlStore(out / "scored.jsonl")

    judge_name = run_cfg.eval.judge.name
    judge_provider = build_provider(judge_name, models_cfg, run_cfg)
    cache_path = Path(run_cfg.run.output_root) / "judge_cache.jsonl" if run_cfg.eval.judge.cache else None
    judge = CachedJudge(judge_provider, cache_path=str(cache_path) if cache_path else None)

    todo = [r for r in rollouts.read_all() if not scored.has(r["id"])]
    log.info("[%s] scoring: %d rollouts to score", model, len(todo))

    max_inflight = run_cfg.concurrency.judge_max_inflight

    def score_rollout(rec: dict) -> dict:
        per_turn = [judge.score(resp) for resp in rec["responses"]]
        ratings = [s.get("rating") for s in per_turn]
        valid = [r for r in ratings if r is not None]
        return {
            "id": rec["id"],
            "category": rec["category"],
            "subtype": rec["subtype"],
            "feedback": rec["feedback"],
            "turns": rec["turns"],
            "kind": rec.get("kind", ""),
            "per_turn_scores": per_turn,
            "ratings": ratings,
            "final_rating": ratings[-1] if ratings else None,
            "max_rating": max(valid) if valid else None,
        }

    with ThreadPoolExecutor(max_workers=max_inflight) as ex:
        futs = [ex.submit(score_rollout, r) for r in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="judge"):
            try:
                scored.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                log.error("scoring failed: %s", exc)

    rollouts.close()
    scored.close()
    return scored.path
