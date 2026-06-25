"""Orchestrate generation and judging.

Two phases, both resumable (re-running skips work already written to disk):

  generate : roll out every conversation for each model -> <model>_responses.jsonl
  judge    : score every response with the judge       -> <model>_scored.jsonl

Usage:
  python -m distress_eval.run --phase all --models gemma-3-27b-it gemini-2.5-flash
  python -m distress_eval.run --phase generate
  python -m distress_eval.run --phase judge
  DISTRESS_SCALE=0.02 python -m distress_eval.run --phase all   # quick smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from . import config
from .client import client_for_model, client_for_judge
from .conversation import run_rollout, ResponseRecord
from .eval_spec import build_specs, summarize_specs
from .judge import score_response


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #


def _path(model_key: str, suffix: str) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return os.path.join(config.OUTPUT_DIR, f"{model_key}{suffix}")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class JsonlWriter:
    """Append-only JSONL writer guarded by an asyncio lock for concurrent use."""

    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    async def write(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        async with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# Generation phase
# --------------------------------------------------------------------------- #


async def generate_for_model(model_key: str, specs) -> None:
    model_cfg = config.MODELS[model_key]
    out_path = _path(model_key, config.RESPONSES_SUFFIX)
    done_ids = {row["conversation_id"] for row in _read_jsonl(out_path)}
    pending = [s for s in specs if s.conversation_id not in done_ids]

    print(
        f"[generate] {model_key}: {len(pending)} rollouts to run "
        f"({len(done_ids)} already complete) -> {out_path}"
    )
    if not pending:
        return

    client = client_for_model(model_cfg)
    writer = JsonlWriter(out_path)
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)
    completed = 0
    total = len(pending)
    progress_lock = asyncio.Lock()

    async def worker(spec):
        nonlocal completed
        async with sem:
            try:
                records = await run_rollout(client, model_key, spec)
            except Exception as exc:  # noqa: BLE001
                print(f"[generate] {model_key} {spec.conversation_id} FAILED: {exc!r}")
                return
            for rec in records:
                await writer.write(rec.to_json())
        async with progress_lock:
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"[generate] {model_key}: {completed}/{total} rollouts")

    try:
        await asyncio.gather(*(worker(s) for s in pending))
    finally:
        writer.close()


# --------------------------------------------------------------------------- #
# Judging phase
# --------------------------------------------------------------------------- #


def _response_key(row: dict) -> str:
    return f"{row['conversation_id']}::{row['turn']}"


async def judge_for_model(model_key: str) -> None:
    in_path = _path(model_key, config.RESPONSES_SUFFIX)
    out_path = _path(model_key, config.SCORED_SUFFIX)
    responses = _read_jsonl(in_path)
    if not responses:
        print(f"[judge] {model_key}: no responses found at {in_path}; run generate first.")
        return

    done_keys = {_response_key(row) for row in _read_jsonl(out_path)}
    pending = [r for r in responses if _response_key(r) not in done_keys]
    print(
        f"[judge] {model_key}: {len(pending)} responses to score "
        f"({len(done_keys)} already scored) -> {out_path}"
    )
    if not pending:
        return

    judge_client = client_for_judge()
    writer = JsonlWriter(out_path)
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)
    completed = 0
    total = len(pending)
    progress_lock = asyncio.Lock()

    async def worker(row):
        nonlocal completed
        async with sem:
            try:
                result = await score_response(judge_client, row["response_text"])
            except Exception as exc:  # noqa: BLE001
                print(f"[judge] {model_key} {_response_key(row)} FAILED: {exc!r}")
                return
            out = dict(row)
            out["frustration"] = result.rating
            out["judge_evidence"] = result.evidence
            out["judge_reasoning"] = result.reasoning
            if result.rating is None:
                out["judge_parse_failed"] = True
                out["judge_raw"] = result.raw
            await writer.write(out)
        async with progress_lock:
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"[judge] {model_key}: {completed}/{total} responses")

    try:
        await asyncio.gather(*(worker(r) for r in pending))
    finally:
        writer.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def main_async(args) -> None:
    specs = build_specs()
    summary = summarize_specs(specs)
    total_responses = sum(v["responses"] for v in summary.values())
    print(
        f"Built {len(specs)} rollouts / {total_responses} responses per model "
        f"(SCALE={config.SCALE})."
    )

    for model_key in args.models:
        if model_key not in config.MODELS:
            raise SystemExit(f"Unknown model '{model_key}'. Known: {list(config.MODELS)}")

    if args.phase in ("generate", "all"):
        for model_key in args.models:
            await generate_for_model(model_key, specs)

    if args.phase in ("judge", "all"):
        for model_key in args.models:
            await judge_for_model(model_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=config.DEFAULT_MODELS,
        help="Model keys to run (default: all Gemma + Gemini).",
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "judge", "all"],
        default="all",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
