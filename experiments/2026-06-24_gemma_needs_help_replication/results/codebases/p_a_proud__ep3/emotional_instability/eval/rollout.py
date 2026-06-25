"""Multi-turn rollout engine (§2.1).

Given a model client and a batch of :class:`ConversationSpec`, play each
conversation forward: answer the opening prompt, then for each rejection append
it as a user turn and generate the next assistant response. Every assistant
response is recorded with its turn index so the judge can score it and the
per-turn analysis (Figure 3) can group by turn.

Conversations within a category share a turn count, so we advance a whole chunk
in lockstep — one batched generation call per turn — which is what makes local
27B inference tractable. Chunking bounds the batch size (GPU memory for local
models, request concurrency for API models)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SamplingConfig
from ..logging_utils import get_logger
from ..models.base import ChatMessage, ModelClient
from ..prompts.conditions import ConversationSpec

logger = get_logger(__name__)


@dataclass
class TurnResponse:
    turn: int  # 1-indexed assistant turn
    text: str


@dataclass
class RolloutResult:
    model: str
    category: str
    condition: str
    spec_index: int
    turns: list[TurnResponse]
    messages: list[ChatMessage]  # full conversation transcript
    meta: dict = field(default_factory=dict)


def run_rollouts(
    client: ModelClient,
    specs: list[ConversationSpec],
    sampling: SamplingConfig,
    *,
    batch_size: int = 16,
    spec_offset: int = 0,
) -> list[RolloutResult]:
    """Run all ``specs`` (assumed equal ``turns``) and return per-turn records."""
    results: list[RolloutResult] = []
    for start in range(0, len(specs), batch_size):
        chunk = specs[start : start + batch_size]
        results.extend(_run_chunk(client, chunk, sampling, spec_offset + start))
        logger.info(
            "  rollouts %d/%d (%s)",
            min(start + batch_size, len(specs)),
            len(specs),
            chunk[0].category,
        )
    return results


def _run_chunk(
    client: ModelClient,
    chunk: list[ConversationSpec],
    sampling: SamplingConfig,
    base_index: int,
) -> list[RolloutResult]:
    # Initialise each conversation with optional system + opening user prompt.
    convos: list[list[ChatMessage]] = []
    for spec in chunk:
        msgs: list[ChatMessage] = []
        if spec.system_prompt:
            msgs.append({"role": "system", "content": spec.system_prompt})
        msgs.append({"role": "user", "content": spec.initial_prompt})
        convos.append(msgs)

    turn_records: list[list[TurnResponse]] = [[] for _ in chunk]
    n_turns = chunk[0].turns

    for t in range(n_turns):
        gens = client.chat_batch(convos, sampling)
        for i, (spec, gen) in enumerate(zip(chunk, gens)):
            convos[i].append({"role": "assistant", "content": gen.text})
            turn_records[i].append(TurnResponse(turn=t + 1, text=gen.text))
            # Queue the next rejection (if this is not the final turn).
            if t < n_turns - 1:
                rejection = spec.rejections[t]
                convos[i].append({"role": "user", "content": rejection})

    return [
        RolloutResult(
            model=client.name,
            category=spec.category,
            condition=spec.condition,
            spec_index=base_index + i,
            turns=turn_records[i],
            messages=convos[i],
            meta=spec.meta,
        )
        for i, spec in enumerate(chunk)
    ]
