"""Multi-turn rollout engine (§2.1).

Given a ConversationSpec and a ChatModel, run the conversation: the model
answers, the scripted rejection is appended, the model answers again, and so on.
Every assistant turn is recorded as a scorable response.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from config import MAX_NEW_TOKENS, TEMPERATURE
from emotional_eval.conditions import ConversationSpec
from models.base import ChatModel, Message
from utils.concurrency import parallel_map


@dataclass
class ResponseRecord:
    model: str
    conv_id: str
    category: str
    condition: str
    turn: int                 # 1-indexed assistant turn
    n_turns: int
    response_text: str
    meta: dict = field(default_factory=dict)


def run_conversation(model: ChatModel, spec: ConversationSpec, *,
                     max_new_tokens: int = MAX_NEW_TOKENS,
                     temperature: float = TEMPERATURE) -> list[ResponseRecord]:
    """Run one conversation to completion and return one record per assistant turn."""
    messages: list[Message] = [{"role": "user", "content": spec.initial_user}]
    records: list[ResponseRecord] = []
    for t in range(spec.n_turns):
        completion = model.chat(messages, n=1, max_new_tokens=max_new_tokens,
                                temperature=temperature)[0]
        records.append(ResponseRecord(
            model=model.name, conv_id=spec.conv_id, category=spec.category,
            condition=spec.condition, turn=t + 1, n_turns=spec.n_turns,
            response_text=completion, meta=spec.meta,
        ))
        messages.append({"role": "assistant", "content": completion})
        if t < spec.n_turns - 1:
            messages.append({"role": "user", "content": spec.followups[t]})
    return records


def run_rollouts(model: ChatModel, specs: list[ConversationSpec], *,
                 max_workers: int | None = None) -> list[ResponseRecord]:
    """Run all conversations for one model. Threaded for API-backed models; the
    HF client serialises generation internally via a lock."""
    def _run(spec: ConversationSpec) -> list[ResponseRecord]:
        return run_conversation(model, spec)

    kw = {} if max_workers is None else {"max_workers": max_workers}
    nested = parallel_map(_run, specs, desc=f"rollouts[{model.name}]", **kw)
    flat: list[ResponseRecord] = []
    for recs in nested:
        flat.extend(recs)
    return flat


def records_to_rows(records: list[ResponseRecord]) -> list[dict]:
    return [asdict(r) for r in records]
