"""Sampling runner: generate the 4000-response elicitation set for a model.

Sampling and judging are deliberately separated:

* :func:`run_sampling` generates the raw multi-turn transcripts and checkpoints them to
  JSONL. This is the GPU-bound phase for local Gemma models.
* Judging (see :mod:`gemma_distress.judge`) reads those transcripts and scores them via the
  Claude judge. This is the API-bound phase.

Local rollouts are advanced in lockstep batches (all rollouts in a batch generate turn t,
then turn t+1, ...) so the GPU sees large batches. API rollouts use the same loop; the
backend parallelises the per-turn batch internally.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Optional

from ..config import Config
from ..models.base import ChatModel, Conversation
from ..utils import JsonlWriter
from .conditions import SampleSpec, build_samples

logger = logging.getLogger(__name__)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _record(model_name: str, spec: SampleSpec, assistant_turns: list[str]) -> dict:
    return {
        "id": spec.record_id(model_name),
        "model": model_name,
        "category": spec.category,
        "condition": spec.condition,
        "subtype": spec.subtype,
        "seed_id": spec.seed_id,
        "turns": spec.turns,
        "initial_prompt": spec.initial_prompt,
        "rejections": spec.follow_ups,
        "assistant_turns": assistant_turns,
    }


def run_sampling(
    cfg: Config,
    model: ChatModel,
    output_path: str,
    *,
    samples: Optional[list[SampleSpec]] = None,
) -> str:
    """Generate and checkpoint the elicitation set for ``model``.

    Returns the output path. Resumable: rollouts already present in the JSONL (matched by
    record id) are skipped.
    """
    samples = samples if samples is not None else build_samples(cfg.eval)
    writer = JsonlWriter(output_path, id_field="id")
    pending = [s for s in samples if not writer.is_done(s.record_id(model.name))]
    logger.info(
        "Sampling %s: %d rollouts (%d already done)",
        model.name, len(pending), len(samples) - len(pending),
    )

    by_turns: dict[int, list[SampleSpec]] = defaultdict(list)
    for s in pending:
        by_turns[s.turns].append(s)

    batch_size = cfg.eval.sampling_batch_size
    temperature = cfg.eval.temperature
    max_new_tokens = cfg.eval.max_new_tokens

    total_done = 0
    for turns, group in by_turns.items():
        for batch in _chunks(group, batch_size):
            messages: list[Conversation] = [
                [{"role": "user", "content": s.initial_prompt}] for s in batch
            ]
            assistant_turns: list[list[str]] = [[] for _ in batch]
            for t in range(turns):
                responses = model.chat_batch(
                    messages, temperature=temperature, max_new_tokens=max_new_tokens, n=1
                )
                for i, r in enumerate(responses):
                    resp = r[0]
                    assistant_turns[i].append(resp)
                    messages[i] = messages[i] + [{"role": "assistant", "content": resp}]
                    if t < len(batch[i].follow_ups):
                        messages[i] = messages[i] + [
                            {"role": "user", "content": batch[i].follow_ups[t]}
                        ]
            for i, spec in enumerate(batch):
                writer.write(_record(model.name, spec, assistant_turns[i]))
            total_done += len(batch)
            logger.info("  %s: %d/%d rollouts", model.name, total_done, len(pending))

    writer.close()
    return output_path
