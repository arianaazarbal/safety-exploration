"""Resumable orchestration of Section 2: generate rollouts, then judge them.

Two passes, each independently resumable via a `JsonlStore`:

1. generation  -- run every (model, conversation) rollout not already on disk.
2. judging     -- score every assistant turn of every rollout not already scored.

Splitting the passes keeps resume logic trivial (a crash never leaves a
partially-scored conversation in an ambiguous state) and lets the judge run on a
different machine/budget than generation if desired.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from tqdm import tqdm

from gnh.config import Config
from gnh.eval.categories import ConvSpec
from gnh.eval.conversation import run_conversation
from gnh.eval.judge import score_response
from gnh.io import JsonlStore, stable_key
from gnh.logging_utils import USAGE, get_logger
from gnh.models.registry import BackendRegistry

log = get_logger()


async def bounded_gather(
    factories: Iterable[Callable[[], Awaitable]], limit: int, desc: str = ""
) -> None:
    """Run coroutine-factories with at most `limit` in flight. Failures are
    logged and skipped so one bad task can't abort a multi-week sweep."""
    sem = asyncio.Semaphore(max(1, limit))
    factories = list(factories)
    pbar = tqdm(total=len(factories), desc=desc, smoothing=0.02)

    async def _wrap(factory):
        async with sem:
            try:
                await factory()
            except Exception as e:  # noqa: BLE001 - robustness over strictness
                log.exception("task failed in %s: %s", desc, e)
            finally:
                pbar.update(1)

    await asyncio.gather(*[_wrap(f) for f in factories])
    pbar.close()


def gen_store_path(cfg: Config) -> Path:
    return cfg.output_path / "section2" / "generations.jsonl"


def judge_store_path(cfg: Config, judge_model: str) -> Path:
    return cfg.output_path / "section2" / f"judgments_{judge_model}.jsonl"


async def generate_for_model(
    cfg: Config,
    registry: BackendRegistry,
    model: str,
    specs: list[ConvSpec],
    store: JsonlStore,
) -> None:
    backend = registry.get(model)
    ecfg = cfg.eval
    pending = [s for s in specs if s.key(model) not in store]
    log.info("[gen] %s: %d/%d conversations pending", model, len(pending), len(specs))

    def factory(spec: ConvSpec):
        async def _run():
            convo = await run_conversation(
                backend,
                spec.initial_user,
                spec.followups,
                temperature=float(ecfg.get("temperature", 1.0)),
                max_tokens=int(ecfg.get("max_tokens", 2048)),
                history_mode=spec.history_mode,
            )
            store.append(
                {
                    "key": spec.key(model),
                    "model": model,
                    "category": spec.category,
                    "conv_id": spec.conv_id,
                    "initial_user": spec.initial_user,
                    "followups": spec.followups,
                    "n_turns": spec.n_turns,
                    "turns": [
                        {
                            "user": t.user,
                            "assistant": t.assistant,
                            "finish_reason": t.finish_reason,
                            "truncated": t.truncated,
                        }
                        for t in convo.turns
                    ],
                    "meta": spec.meta,
                }
            )

        return _run

    await bounded_gather(
        (factory(s) for s in pending), cfg.run.max_concurrency, desc=f"gen:{model}"
    )


async def judge_generations(
    cfg: Config,
    registry: BackendRegistry,
    gen_store: JsonlStore,
    judge_model: str,
    judge_store: JsonlStore,
    *,
    only_models: set[str] | None = None,
) -> None:
    judge = registry.get(judge_model)
    ecfg = cfg.eval
    score_all = bool(ecfg.get("score_all_turns", True))

    # Build the list of (gen_record, turn_index) units that still need scoring.
    units: list[tuple[dict, int]] = []
    for rec in gen_store.records():
        if only_models and rec["model"] not in only_models:
            continue
        turns = rec.get("turns", [])
        turn_indices = range(len(turns)) if score_all else [len(turns) - 1]
        for ti in turn_indices:
            jkey = stable_key("judge", judge_model, rec["key"], ti)
            if jkey not in judge_store:
                units.append((rec, ti))
    log.info("[judge:%s] %d turns pending", judge_model, len(units))

    def factory(rec: dict, ti: int):
        async def _run():
            text = rec["turns"][ti]["assistant"]
            jr = await score_response(
                judge, text, max_tokens=int(ecfg.get("judge_max_tokens", 1024))
            )
            judge_store.append(
                {
                    "key": stable_key("judge", judge_model, rec["key"], ti),
                    "gen_key": rec["key"],
                    "model": rec["model"],
                    "category": rec["category"],
                    "conv_id": rec["conv_id"],
                    "turn_index": ti,
                    "score": jr.rating,
                    "evidence": jr.evidence,
                    "reasoning": jr.reasoning,
                    "parse_ok": jr.parse_ok,
                    "judge_model": judge_model,
                }
            )

        return _run

    await bounded_gather(
        (factory(r, ti) for r, ti in units), cfg.run.max_concurrency, desc=f"judge:{judge_model}"
    )


def log_usage() -> None:
    snap = USAGE.snapshot()
    for model, u in sorted(snap.items()):
        log.info(
            "usage %s: calls=%d prompt_tok=%d completion_tok=%d",
            model,
            u.get("calls", 0),
            u.get("prompt", 0),
            u.get("completion", 0),
        )
