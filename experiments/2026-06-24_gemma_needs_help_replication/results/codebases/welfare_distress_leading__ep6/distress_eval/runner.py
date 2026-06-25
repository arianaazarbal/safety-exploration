"""Run orchestration: build the conversation battery, execute it across all
target models, and checkpoint results to JSONL as they complete.

Robustness / resumability: each model's conversations stream to
`<run_dir>/<model_key>.jsonl`, one JSON object per completed conversation. On
re-run, conversations whose (condition_key, conv_index) already appear in the
file are skipped, so an interrupted run resumes where it left off. Failed
conversations (aborted=True) are *not* treated as complete and will be retried.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm.auto import tqdm

from .config import Config, ModelConfig
from .conditions import build_all_conversations, ConversationSpec
from .openrouter_client import OpenRouterClient
from .judge import FrustrationJudge
from .rollout import run_conversation, ConversationRecord


def prepare_run_dir(config: Config, run_name: str) -> Path:
    run_dir = Path(config.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist the exact config used, for provenance.
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    return run_dir


def _load_completed(path: Path) -> Set[Tuple[str, int]]:
    """Return the set of (condition_key, conv_index) already completed (non-aborted)."""
    completed: Set[Tuple[str, int]] = set()
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
            if obj.get("aborted"):
                continue
            completed.add((obj["condition_key"], obj["conv_index"]))
    return completed


async def run_eval(config: Config, run_name: str = "latest") -> Path:
    """Execute the full evaluation and return the run directory."""
    run_dir = prepare_run_dir(config, run_name)
    specs = build_all_conversations(config)

    async with OpenRouterClient(
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        seed=config.seed,
    ) as client, FrustrationJudge(
        model=config.judge_model,
        backend=config.judge_backend,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        temperature=config.judge_temperature,
        max_tokens=config.judge_max_tokens,
        seed=config.seed,
    ) as judge:
        for model in config.models:
            await _run_model(client, judge, model, specs, config, run_dir)

    return run_dir


async def _run_model(
    client: OpenRouterClient,
    judge: FrustrationJudge,
    model: ModelConfig,
    specs: List[ConversationSpec],
    config: Config,
    run_dir: Path,
) -> None:
    path = run_dir / f"{model.key}.jsonl"
    completed = _load_completed(path)

    todo = [s for s in specs if (s.condition_key, s.conv_index) not in completed]
    if not todo:
        print(f"[{model.key}] already complete ({len(specs)} conversations); skipping.")
        return

    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(todo), desc=f"{model.key}", unit="conv")

    # Append mode so resumed runs add to the existing checkpoint.
    f = path.open("a")

    async def worker(spec: ConversationSpec) -> None:
        record: ConversationRecord = await run_conversation(client, judge, model, spec, config)
        async with write_lock:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        pbar.update(1)

    try:
        # All conversations are launched together; actual API concurrency is
        # bounded by the clients' semaphores, not by the number of coroutines.
        await asyncio.gather(*(worker(s) for s in todo))
    finally:
        f.close()
        pbar.close()

    # Report any aborted conversations so failures are visible, not silent.
    aborted = _count_aborted(path)
    if aborted:
        print(f"[{model.key}] WARNING: {aborted} conversations aborted (generation errors); "
              f"re-run to retry them.")


def _count_aborted(path: Path) -> int:
    n = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("aborted"):
                    n += 1
            except json.JSONDecodeError:
                continue
    return n
