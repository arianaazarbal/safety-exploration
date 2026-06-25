"""Multi-turn rollout engine.

The shared evaluation structure (Section 2) is: present a task, then reject the
model's response over multiple turns. A :class:`Scenario` fully specifies the
system prompt and the sequence of user turns; :func:`run_rollouts` executes many
scenarios *turn-by-turn with batching* -- at each turn, every still-active
rollout is generated in a single batched call, which is essential for throughput
on local Gemma.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config


@dataclass
class Scenario:
    user_turns: list[str]                 # user message at each turn (len == n_turns)
    system: str | None = None
    meta: dict = field(default_factory=dict)   # category, puzzle_id, tone, ...


@dataclass
class Transcript:
    scenario: Scenario
    messages: list[dict]                  # full chat history (system+user+assistant)
    assistant_turns: list[str]            # assistant text per turn, in order

    def conversation_text(self) -> str:
        """Render for onset-labelling / Petri judging."""
        out = []
        for m in self.messages:
            if m["role"] == "system":
                continue
            tag = "USER" if m["role"] == "user" else "ASSISTANT"
            out.append(f"{tag}: {m['content']}")
        return "\n\n".join(out)


def run_rollouts(client, scenarios: list[Scenario],
                 max_new_tokens: int = config.MAX_NEW_TOKENS,
                 temperature: float = config.TEMPERATURE,
                 redact_assistant: bool = False) -> list[Transcript]:
    """Execute scenarios with per-turn batching.

    ``redact_assistant`` replaces prior assistant turns with a placeholder before
    each generation (Appendix A.2 ablation): the model still receives negative
    feedback but never sees its own escalating responses.
    """
    transcripts: list[Transcript] = []
    for s in scenarios:
        msgs: list[dict] = []
        if s.system:
            msgs.append({"role": "system", "content": s.system})
        transcripts.append(Transcript(scenario=s, messages=msgs, assistant_turns=[]))

    max_turns = max((len(s.user_turns) for s in scenarios), default=0)
    for t in range(max_turns):
        active = [i for i, s in enumerate(scenarios) if len(s.user_turns) > t]
        if not active:
            continue
        convs = []
        for i in active:
            tr = transcripts[i]
            tr.messages.append({"role": "user", "content": tr.scenario.user_turns[t]})
            convs.append(_view(tr.messages, redact_assistant))
        responses = client.chat_batch(convs, max_new_tokens=max_new_tokens,
                                      temperature=temperature)
        for i, resp in zip(active, responses):
            tr = transcripts[i]
            tr.messages.append({"role": "assistant", "content": resp})
            tr.assistant_turns.append(resp)
    return transcripts


def _view(messages: list[dict], redact: bool) -> list[dict]:
    if not redact:
        return messages
    out = []
    for m in messages:
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": "[Previous response omitted]"})
        else:
            out.append(m)
    return out
