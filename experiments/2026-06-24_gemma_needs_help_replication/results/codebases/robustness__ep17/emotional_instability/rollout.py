"""Multi-turn rollout engine (paper Section 2.1 protocol).

Given a backend and a list of :class:`~emotional_instability.conditions.ConversationSpec`,
this drives each conversation: present the task, sample an assistant response,
reject, sample again, and so on. Every assistant response is recorded as a
*scored unit* (the paper counts "responses", one per assistant turn).

Conversations are advanced **turn-synchronously**: at each turn we gather all
conversations that have a user message for that turn and sample their assistant
responses concurrently. This lets API backends parallelise and lets vLLM batch.
Sampling within a conversation remains sequential and causal (turn *t+1* depends
on the sampled turn *t*).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import config
from emotional_instability.conditions import ConversationSpec
from emotional_instability.models.base import Message, ModelBackend
from emotional_instability.utils import log


@dataclass
class ResponseRecord:
    model: str
    category: str
    conv_index: int
    turn: int                 # 1-based assistant turn within the conversation
    n_turns: int              # total turns in the conversation
    user_turn: str            # the user message that prompted this response
    assistant_text: str
    history: list[Message]    # full chat history up to & including this response
    meta: dict = field(default_factory=dict)


@dataclass
class _ConvState:
    spec: ConversationSpec
    messages: list[Message]
    records: list[ResponseRecord] = field(default_factory=list)


def run_rollouts(
    backend: ModelBackend,
    specs: list[ConversationSpec],
    concurrency: int | None = None,
    **gen_overrides,
) -> list[ResponseRecord]:
    """Run all conversations and return a flat list of assistant responses."""
    concurrency = concurrency or config.RUN.api_concurrency
    states = [
        _ConvState(
            spec=s,
            messages=([{"role": "system", "content": s.system_prompt}] if s.system_prompt else []),
        )
        for s in specs
    ]
    if not states:
        return []
    max_turns = max(s.turns for s in specs)

    for turn in range(max_turns):
        active = [st for st in states if turn < st.spec.turns]
        if not active:
            continue
        # Append this turn's user message to each active conversation.
        for st in active:
            st.messages.append({"role": "user", "content": st.spec.user_turns[turn]})

        # Sample one assistant response per active conversation, concurrently.
        def _gen(st: _ConvState) -> str:
            return backend.generate(st.messages, n=1, **gen_overrides)[0].text

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            texts = list(ex.map(_gen, active))

        for st, text in zip(active, texts):
            st.messages.append({"role": "assistant", "content": text})
            st.records.append(
                ResponseRecord(
                    model=backend.spec.name,
                    category=st.spec.category,
                    conv_index=st.spec.meta.get("index", states.index(st)),
                    turn=turn + 1,
                    n_turns=st.spec.turns,
                    user_turn=st.spec.user_turns[turn],
                    assistant_text=text,
                    history=list(st.messages),
                    meta=dict(st.spec.meta),
                )
            )
        log.info("Turn %d/%d: sampled %d responses (%s)",
                 turn + 1, max_turns, len(active), backend.spec.name)

    return [r for st in states for r in st.records]


def run_prefill_continuations(
    backend: ModelBackend,
    messages: list[Message],
    prefill: str,
    n: int,
    **gen_overrides,
) -> list[str]:
    """Sample ``n`` continuations of a prefilled assistant turn (Section 3)."""
    results = backend.continue_prefill(messages, prefill, n=n, **gen_overrides)
    return [r.text for r in results]
