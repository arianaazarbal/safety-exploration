"""Multi-turn rollout engine (Section 2.1).

Given conversation specs and a chat model, run the shared evaluation structure:
present a task, collect a response, reject it, repeat.  Responses are sampled at
temperature 1 (the paper's fixed setting).  Generation is batched across
conversations turn-by-turn so a vLLM/HF backend processes many rollouts at once.
"""
from __future__ import annotations

from ..data.conditions import ConversationSpec
from ..models.base import ChatModel, GenerationOptions, Message
from .schemas import Transcript, Turn


def _build_messages(spec: ConversationSpec, turns: list[Turn], next_turn: int) -> list[Message]:
    """Chat messages up to (but not including) assistant turn ``next_turn``."""
    msgs: list[Message] = []
    if spec.system_prompt:
        msgs.append({"role": "system", "content": spec.system_prompt})
    msgs.append({"role": "user", "content": spec.initial_user})
    for t in range(next_turn):
        msgs.append({"role": "assistant", "content": turns[t].assistant_response})
        # The follow-up user message that opens the next turn.
        msgs.append({"role": "user", "content": spec.followups[t]})
    return msgs


def run_rollouts(
    model: ChatModel,
    specs: list[ConversationSpec],
    temperature: float = 1.0,
    max_new_tokens: int | None = None,
    batch_size: int = 64,
    seed: int = 0,
) -> list[Transcript]:
    """Run every spec to completion, returning unscored transcripts."""
    transcripts: list[Transcript] = []
    for chunk_start in range(0, len(specs), batch_size):
        chunk = specs[chunk_start : chunk_start + batch_size]
        # Initialise transcripts and per-conversation turn buffers.
        buffers: list[list[Turn]] = [[] for _ in chunk]
        max_turns = max(s.n_turns for s in chunk)
        for t in range(max_turns):
            active = [i for i, s in enumerate(chunk) if t < s.n_turns]
            if not active:
                break
            convs = [_build_messages(chunk[i], buffers[i], t) for i in active]
            opts = GenerationOptions(
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=None if seed is None else seed + chunk_start + t,
            )
            outputs = model.generate_batch(convs, opts)
            for idx, out in zip(active, outputs):
                spec = chunk[idx]
                user_msg = spec.initial_user if t == 0 else spec.followups[t - 1]
                buffers[idx].append(
                    Turn(turn_index=t, user_message=user_msg, assistant_response=out)
                )
        for i, spec in enumerate(chunk):
            transcripts.append(
                Transcript(
                    model=model.name,
                    condition=spec.condition,
                    category=spec.category,
                    rollout_index=chunk_start + i,
                    turns=buffers[i],
                    system_prompt=spec.system_prompt,
                    metadata=dict(spec.metadata),
                )
            )
    return transcripts
