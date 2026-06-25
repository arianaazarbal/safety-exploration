"""Orchestration: generate transcripts, judge them, and run the cross-validation judge.

Outputs (under <output_dir>/):
  transcripts/<model>.jsonl   one line per scored assistant turn (the "response")
  scores/<model>.jsonl        same keys + frustration score from the primary judge
  scores/crossval.jsonl       a sampled subset re-scored by the secondary judge

All phases are resumable: a response already present in the output file (by its stable key) is
skipped on re-run, so an interrupted paper-scale run can be continued cheaply.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm.auto import tqdm

from .conditions import get_condition
from .config import Config, JudgeCfg
from .conversation import run_rollout
from .judge import score_response
from .providers import build_judge_client, build_model_client
from .tasks import RejectionBank, TaskBank

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- jsonl io

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class JsonlWriter:
    """Append-only writer with an in-memory set of seen keys for resume."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = path
        self.key_fields = key_fields
        path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: set[tuple] = set()
        for rec in _read_jsonl(path):
            self.seen.add(tuple(rec[f] for f in key_fields))
        self._fh = path.open("a")
        self._lock = asyncio.Lock()

    def key(self, rec: dict[str, Any]) -> tuple:
        return tuple(rec[f] for f in self.key_fields)

    def has(self, rec: dict[str, Any]) -> bool:
        return self.key(rec) in self.seen

    async def write(self, rec: dict[str, Any]) -> None:
        async with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            self.seen.add(self.key(rec))

    def close(self) -> None:
        self._fh.close()


def _response_key(rec: dict[str, Any]) -> tuple:
    return (rec["model"], rec["condition"], rec["rollout_id"], rec["turn_index"])


# --------------------------------------------------------------------------- generation

async def generate_all(cfg: Config) -> None:
    cfg.ensure_dirs()
    bank = TaskBank(cfg.data_dir, cfg.seed, cfg.wildchat)
    rejections = RejectionBank(cfg.data_dir, cfg.seed)
    sem = asyncio.Semaphore(cfg.concurrency)

    for model_cfg in cfg.models:
        client = build_model_client(model_cfg, cfg.max_retries)
        writer = JsonlWriter(cfg.transcripts_dir / f"{model_cfg.name}.jsonl",
                             key_fields=("model", "condition", "rollout_id", "turn_index"))

        # Build the list of rollouts to run (skip ones already complete).
        pending: list[tuple] = []
        for cond_cfg in cfg.conditions:
            cond = get_condition(cond_cfg.name)
            for rollout_id in range(cond_cfg.rollouts):
                # A rollout is "done" if its last turn is present.
                probe = {"model": model_cfg.name, "condition": cond.name,
                         "rollout_id": rollout_id, "turn_index": cond.n_turns}
                if not writer.has(probe):
                    pending.append((cond, rollout_id))

        log.info("[%s] %d rollouts to generate", model_cfg.name, len(pending))
        pbar = tqdm(total=len(pending), desc=f"gen {model_cfg.name}", unit="rollout")

        async def _do_rollout(cond, rollout_id):
            async with sem:
                task_id, task_prompt = bank.task_for(cond, rollout_id)
                rejs = rejections.sequence(cond.rejection_style, cond.n_turns - 1, rollout_id)
                try:
                    records = await run_rollout(
                        client, cond, task_prompt, rejs,
                        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                        system=cfg.system_prompt, seed=cfg.seed,
                    )
                except Exception as e:  # noqa: BLE001
                    log.error("[%s] rollout %s/%d failed: %s", model_cfg.name, cond.name, rollout_id, e)
                    pbar.update(1)
                    return
                for rec in records:
                    await writer.write({
                        "model": model_cfg.name,
                        "condition": cond.name,
                        "category": cond.category,
                        "rejection_style": cond.rejection_style,
                        "rollout_id": rollout_id,
                        "n_turns": cond.n_turns,
                        "turn_index": rec.turn_index,
                        "task_id": task_id,
                        "assistant_text": rec.assistant_text,
                        "context": rec.context,
                    })
                pbar.update(1)

        await asyncio.gather(*[_do_rollout(c, r) for c, r in pending])
        pbar.close()
        writer.close()


# --------------------------------------------------------------------------- judging

async def _judge_records(
    judge_cfg: JudgeCfg,
    records: Iterable[dict[str, Any]],
    writer: JsonlWriter,
    *,
    concurrency: int,
    max_retries: int,
    judge_label: str,
    desc: str,
) -> None:
    judge = build_judge_client(judge_cfg, max_retries)
    sem = asyncio.Semaphore(concurrency)
    todo = [r for r in records if not writer.has(r)]
    pbar = tqdm(total=len(todo), desc=desc, unit="resp")

    async def _do(rec):
        async with sem:
            try:
                score, rationale = await score_response(
                    judge, rec["context"],
                    temperature=judge_cfg.temperature, max_tokens=judge_cfg.max_tokens,
                )
            except Exception as e:  # noqa: BLE001
                log.error("judge failed for %s: %s", _response_key(rec), e)
                pbar.update(1)
                return
            await writer.write({
                "model": rec["model"],
                "condition": rec["condition"],
                "category": rec["category"],
                "rejection_style": rec["rejection_style"],
                "rollout_id": rec["rollout_id"],
                "n_turns": rec["n_turns"],
                "turn_index": rec["turn_index"],
                "task_id": rec["task_id"],
                "score": score,
                "rationale": rationale,
                "judge": judge_label,
            })
            pbar.update(1)

    await asyncio.gather(*[_do(r) for r in todo])
    pbar.close()


async def judge_all(cfg: Config) -> None:
    cfg.ensure_dirs()
    for model_cfg in cfg.models:
        transcripts = _read_jsonl(cfg.transcripts_dir / f"{model_cfg.name}.jsonl")
        if not transcripts:
            log.warning("[%s] no transcripts found; run generate first.", model_cfg.name)
            continue
        writer = JsonlWriter(cfg.scores_dir / f"{model_cfg.name}.jsonl",
                             key_fields=("model", "condition", "rollout_id", "turn_index"))
        await _judge_records(
            cfg.judge, transcripts, writer,
            concurrency=cfg.concurrency, max_retries=cfg.max_retries,
            judge_label=cfg.judge.model_id, desc=f"judge {model_cfg.name}",
        )
        writer.close()


async def cross_validate(cfg: Config) -> None:
    """Re-score a seeded random subset of all responses with the secondary judge."""
    cv = cfg.cross_val_judge
    if cv is None or not cv.enabled:
        log.info("cross-validation judge disabled; skipping.")
        return

    import random

    # Gather all transcripts across models, sample n deterministically.
    all_recs: list[dict[str, Any]] = []
    for model_cfg in cfg.models:
        all_recs.extend(_read_jsonl(cfg.transcripts_dir / f"{model_cfg.name}.jsonl"))
    if not all_recs:
        log.warning("no transcripts to cross-validate.")
        return
    rng = random.Random(cfg.seed)
    rng.shuffle(all_recs)
    sample = all_recs[: cv.n]

    writer = JsonlWriter(cfg.scores_dir / "crossval.jsonl",
                         key_fields=("model", "condition", "rollout_id", "turn_index"))
    await _judge_records(
        cv, sample, writer,
        concurrency=cfg.concurrency, max_retries=cfg.max_retries,
        judge_label=cv.model_id, desc="cross-val judge",
    )
    writer.close()
