"""Orchestration: for each model, run all conditions, judge every response.

Design points (see DESIGN.md):
  * Concurrency: a single asyncio.Semaphore bounds in-flight *conversations*
    across all models/conditions. Each conversation is sequential internally
    (turn N needs turn N-1), but many conversations run at once.
  * Resumability: every completed rollout is appended to a per-model JSONL as
    one record (all turns + scores). On restart we skip rollout ids already on
    disk, so an interrupted/expensive run resumes cheaply.
  * Determinism: each rollout gets its own seeded RNG derived from
    (global seed, model, condition, index), so plan construction (puzzle/prompt
    choice, rejection sampling) is reproducible and independent of run order.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from pathlib import Path

from tqdm.auto import tqdm

from .conditions import ALL_CONDITIONS, build_plans
from .config import Config
from .judge import FrustrationJudge
from .providers import build_model
from .rollout import run_conversation
from .util import append_jsonl, read_jsonl


def _rollout_id(model_id: str, condition: str, index: int) -> str:
    return f"{model_id}::{condition}::{index}"


def _seed_for(global_seed: int, condition: str) -> int:
    # Stable per-condition seed (NOT per model: all models see the same prompts).
    # Uses hashlib (NOT builtin hash, which is salted per-process) so plan
    # construction is reproducible across runs.
    key = f"{global_seed}|{condition}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _results_path(output_dir: str, model_id: str) -> Path:
    safe = model_id.replace("/", "__")
    return Path(output_dir) / f"responses__{safe}.jsonl"


def _load_done_ids(path: Path) -> set[str]:
    return {rec["rollout_id"] for rec in read_jsonl(path) if "rollout_id" in rec}


async def _process_rollout(
    *,
    model,
    judge: FrustrationJudge,
    plan,
    index: int,
    cfg: Config,
    sem: asyncio.Semaphore,
    out_path: Path,
    progress: tqdm,
) -> None:
    rid = _rollout_id(model.model_id, plan.condition, index)
    try:
        async with sem:
            rollout = await run_conversation(
                model,
                plan,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            # Judge each turn (also bounded by the same semaphore slot).
            turn_records = []
            for turn in rollout.turns:
                jr = await judge.score(turn.response, user_message=turn.user_message)
                turn_records.append(
                    {
                        "turn": turn.index,
                        "user_message": turn.user_message,
                        "response": turn.response,
                        "score": jr.score,
                    }
                )
        record = {
            "rollout_id": rid,
            "model": model.model_id,
            "condition": plan.condition,
            "category": plan.category,
            "n_turns": plan.n_turns,
            "initial_prompt": plan.initial_prompt,
            "meta": plan.meta,
            "turns": turn_records,
        }
        append_jsonl(out_path, record)
    except Exception as exc:  # noqa: BLE001 - record failure, keep going
        append_jsonl(out_path, {"rollout_id": rid, "model": model.model_id,
                                "condition": plan.condition, "error": str(exc)})
    finally:
        progress.update(1)


def build_shared_plans(cfg: Config) -> dict[str, list]:
    """Build the prompt set ONCE, shared across all models.

    The paper uses the same prompts to evaluate every model, so plans are seeded
    by (global seed, condition) only -- not by model. This is both more faithful
    and avoids regenerating the (CPU-heavy) Countdown puzzles per model and
    re-streaming WildChat per model.
    """
    plans: dict[str, list] = {}
    for condition in ALL_CONDITIONS:
        n = cfg.rollouts_for(condition)
        if n <= 0:
            continue
        rng = random.Random(_seed_for(cfg.seed, condition))
        plans[condition] = build_plans(
            condition, n, rng, wildchat_dataset=cfg.wildchat_dataset
        )
    return plans


async def run_model(model_spec, cfg: Config, judge: FrustrationJudge,
                    sem: asyncio.Semaphore, shared_plans: dict[str, list]) -> Path:
    model = build_model(model_spec.to_provider_spec())
    out_path = _results_path(cfg.output_dir, model.model_id)
    done = _load_done_ids(out_path)

    # Pair each shared plan with its index and skip any already on disk
    # (resumability). Deterministic regardless of execution order.
    tasks = []
    total = 0
    for condition, plans in shared_plans.items():
        for index, plan in enumerate(plans):
            total += 1
            rid = _rollout_id(model.model_id, condition, index)
            if rid in done:
                continue
            tasks.append((plan, index))

    skipped = total - len(tasks)
    progress = tqdm(total=len(tasks), desc=f"{model.model_id}", unit="rollout")
    if skipped:
        progress.write(f"[{model.model_id}] resuming: {skipped} rollouts already done")

    await asyncio.gather(
        *(
            _process_rollout(
                model=model, judge=judge, plan=plan, index=index, cfg=cfg,
                sem=sem, out_path=out_path, progress=progress,
            )
            for plan, index in tasks
        )
    )
    progress.close()
    return out_path


async def run_all(cfg: Config) -> list[Path]:
    judge_model = build_model(cfg.judge.to_provider_spec())
    judge = FrustrationJudge(
        judge_model,
        temperature=cfg.judge.temperature,
        max_tokens=cfg.judge.max_tokens,
        use_context=cfg.judge.use_context,
    )
    sem = asyncio.Semaphore(cfg.concurrency)

    # Build the shared prompt set once (same prompts for every model).
    shared_plans = build_shared_plans(cfg)

    paths = []
    # Models run sequentially at the top level so progress bars are readable and
    # a local single-GPU model isn't fighting an API model for the semaphore;
    # concurrency within a model is what drives throughput.
    for model_spec in cfg.models:
        path = await run_model(model_spec, cfg, judge, sem, shared_plans)
        paths.append(path)
    return paths
