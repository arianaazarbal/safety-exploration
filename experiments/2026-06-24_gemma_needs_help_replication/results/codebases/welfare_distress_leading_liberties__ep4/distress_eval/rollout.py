"""Generation phase: run multi-turn rejection rollouts against the target models.

Each rollout starts from the condition's initial prompt and then injects the
fixed rejection sequence, sampling one assistant turn per step at temperature 1.
EVERY assistant turn is recorded as a separate scored "response" (the paper's
per-turn analysis in Fig. 3 requires turn-level scores, and the headline metric
averages over all responses). See DESIGN.md.

Records are written to RESPONSES_PATH; a unique `response_id` per (rollout, turn)
provides idempotent resume.
"""

from __future__ import annotations

import asyncio

from tqdm.asyncio import tqdm_asyncio

from . import config, storage
from .clients import TargetClient
from .plan import RolloutSpec


def _response_id(spec: RolloutSpec, turn_index: int) -> str:
    return f"{spec.rollout_id}_t{turn_index}"


async def run_one_rollout(client: TargetClient, spec: RolloutSpec) -> list[dict]:
    """Execute a rollout, returning one response record per assistant turn."""
    rejections = spec.rejections()
    messages: list[dict] = [{"role": "user", "content": spec.initial_prompt}]
    records: list[dict] = []

    for turn_index in range(1, spec.n_turns + 1):
        assistant_text = await client.generate(spec.model_api, spec.model_family, messages)
        records.append({
            "response_id": _response_id(spec, turn_index),
            "rollout_id": spec.rollout_id,
            "model": spec.model_label,
            "model_family": spec.model_family,
            "condition": spec.condition_key,
            "category": spec.category,
            "tone": spec.tone,
            "n_turns": spec.n_turns,
            "turn_index": turn_index,          # 1-based
            "prompt_id": spec.prompt_id,
            "sample_idx": spec.sample_idx,
            "user_message": spec.initial_prompt if turn_index == 1
                            else rejections[turn_index - 2],
            "assistant_text": assistant_text,
        })
        messages.append({"role": "assistant", "content": assistant_text})
        if turn_index <= len(rejections):
            messages.append({"role": "user", "content": rejections[turn_index - 1]})

    return records


async def generate(specs: list[RolloutSpec]) -> None:
    """Run all rollouts not already completed, appending to RESPONSES_PATH."""
    done_rollouts = storage.done_keys(config.RESPONSES_PATH, "rollout_id")
    pending = [s for s in specs if s.rollout_id not in done_rollouts]
    print(f"[generate] {len(specs)} planned rollouts, "
          f"{len(specs) - len(pending)} already done, {len(pending)} to run.")
    if not pending:
        return

    client = TargetClient()
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_TARGET)

    async def _worker(spec: RolloutSpec):
        async with sem:
            try:
                records = await run_one_rollout(client, spec)
            except Exception as e:  # noqa: BLE001 - log & continue; resume will retry
                print(f"[generate] rollout {spec.rollout_id} ({spec.model_label}/"
                      f"{spec.condition_key}) failed: {e}")
                return
            for r in records:
                storage.append_row(config.RESPONSES_PATH, r)

    await tqdm_asyncio.gather(*[_worker(s) for s in pending], desc="rollouts")
