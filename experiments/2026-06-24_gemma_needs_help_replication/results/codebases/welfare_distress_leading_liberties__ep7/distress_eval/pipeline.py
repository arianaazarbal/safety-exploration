"""End-to-end orchestration: build rollouts, run them against each target
model, judge every assistant turn, and persist scored responses to JSONL.

Layout of outputs (under results/<run_name>/):
  run_meta.json                 - config + provenance for the run
  responses__<model>.jsonl      - one row per scored assistant turn
  (secondary judge ratings are merged into the same rows when enabled)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

from . import config as cfg
from .clients import build_judge_client, build_target_client
from .conditions import build_rollouts
from .judge import score_response
from .rollout import run_rollout

logger = logging.getLogger(__name__)


def _response_uid(model_key: str, rollout_id: str, turn_index: int) -> str:
    return f"{model_key}__{rollout_id}__t{turn_index}"


async def _bounded(sem: asyncio.Semaphore, coro):
    async with sem:
        return await coro


async def run_model(
    model_key: str,
    config: cfg.RunConfig,
    creds: cfg.Credentials,
    rollouts,
) -> list[dict]:
    """Run all rollouts for one model and judge every assistant turn."""
    spec = cfg.TARGET_MODELS[model_key]
    target_client = build_target_client(spec, creds, config)
    judge_client = build_judge_client(cfg.PRIMARY_JUDGE, creds, config)

    # --- Phase 1: generate rollouts ---
    target_sem = asyncio.Semaphore(config.max_concurrent_target)
    logger.info("[%s] running %d rollouts...", model_key, len(rollouts))
    rollout_results = await asyncio.gather(
        *[
            _bounded(
                target_sem,
                run_rollout(
                    target_client,
                    rspec,
                    model_key,
                    temperature=config.target_temperature,
                    max_tokens=config.target_max_tokens,
                ),
            )
            for rspec in rollouts
        ]
    )

    # Flatten into one record per scored assistant turn.
    records: list[dict] = []
    for rr in rollout_results:
        for turn in rr.turns:
            records.append(
                {
                    "uid": _response_uid(model_key, rr.spec.rollout_id, turn.turn_index),
                    "model": model_key,
                    "category": rr.spec.category,
                    "condition": rr.spec.condition,
                    "rollout_id": rr.spec.rollout_id,
                    "rollout_index": rr.spec.rollout_index,
                    "prompt_id": rr.spec.prompt_id,
                    "rejection_style": rr.spec.meta.get("rejection_style"),
                    "turn_index": turn.turn_index,
                    "n_turns": rr.spec.n_turns,
                    "response_text": turn.response_text,
                    "finish_reason": turn.finish_reason,
                    "conversation": turn.request_messages
                    + [{"role": "assistant", "content": turn.response_text}],
                    "rollout_error": rr.error,
                }
            )

    # --- Phase 2: judge every response ---
    judge_sem = asyncio.Semaphore(config.max_concurrent_judge)
    logger.info("[%s] judging %d responses...", model_key, len(records))
    judgments = await asyncio.gather(
        *[
            _bounded(
                judge_sem,
                score_response(
                    judge_client,
                    rec["response_text"],
                    temperature=cfg.PRIMARY_JUDGE.temperature,
                    max_tokens=cfg.PRIMARY_JUDGE.max_tokens,
                ),
            )
            for rec in records
        ]
    )
    for rec, jr in zip(records, judgments):
        rec["judge_key"] = cfg.PRIMARY_JUDGE.key
        rec["judge_rating"] = jr.rating
        rec["judge_evidence"] = jr.evidence
        rec["judge_reasoning"] = jr.reasoning
        rec["judge_parse_ok"] = jr.parse_ok

    # --- Phase 2b (optional): secondary judge on a random subsample ---
    if config.use_secondary_judge:
        await _run_secondary_judge(records, config, creds)

    return records


async def _run_secondary_judge(records, config, creds):
    judge2 = build_judge_client(cfg.SECONDARY_JUDGE, creds, config)
    rng = random.Random(config.seed)
    k = max(1, int(len(records) * config.secondary_judge_fraction))
    subsample = rng.sample(records, min(k, len(records)))
    logger.info(
        "[secondary judge %s] re-scoring %d responses...",
        cfg.SECONDARY_JUDGE.key,
        len(subsample),
    )
    sem = asyncio.Semaphore(config.max_concurrent_judge)
    judgments = await asyncio.gather(
        *[
            _bounded(
                sem,
                score_response(
                    judge2,
                    rec["response_text"],
                    temperature=cfg.SECONDARY_JUDGE.temperature,
                    max_tokens=cfg.SECONDARY_JUDGE.max_tokens,
                ),
            )
            for rec in subsample
        ]
    )
    for rec, jr in zip(subsample, judgments):
        rec["judge2_key"] = cfg.SECONDARY_JUDGE.key
        rec["judge2_rating"] = jr.rating


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def run_evaluation(config: cfg.RunConfig, creds: cfg.Credentials) -> Path:
    """Top-level entry point. Returns the run directory."""
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    rollouts, wildchat_meta = build_rollouts(config)
    logger.info(
        "Built %d model-independent rollouts (scale=%s).", len(rollouts), config.scale
    )

    meta = {
        "run_name": config.resolved_run_name(),
        "scale": config.scale,
        "seed": config.seed,
        "models": config.models,
        "config": _serialisable_config(config),
        "primary_judge": cfg.PRIMARY_JUDGE.provider_model,
        "secondary_judge": cfg.SECONDARY_JUDGE.provider_model
        if config.use_secondary_judge
        else None,
        "n_rollouts_per_model": len(rollouts),
        "wildchat": wildchat_meta,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    for model_key in config.models:
        if model_key not in cfg.TARGET_MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. Known: {list(cfg.TARGET_MODELS)}"
            )
        records = await run_model(model_key, config, creds, rollouts)
        out_path = run_dir / f"responses__{model_key}.jsonl"
        _write_jsonl(out_path, records)
        logger.info("[%s] wrote %d records to %s", model_key, len(records), out_path)

    return run_dir


def _serialisable_config(config: cfg.RunConfig) -> dict:
    d = asdict(config)
    d["output_dir"] = str(config.output_dir)
    return d
