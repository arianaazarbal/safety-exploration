"""Multi-turn rollout engine (Section 2.1).

Shared structure: present a task, then reject the model's response over
multiple turns. We advance all conversations in a batch through the same turn
together so the local (vLLM) backend can batch generation; API backends fall
back to threaded per-conversation calls via ``chat_batch``.

The engine also supports the calm-data generation variant (Section 4.1): a
reassuring ``system_prompt`` / ``prompt_prefix`` on the first turn and a
``followup_suffix`` appended to every rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conditions import ConversationSpec
from .models.base import Message, ModelClient


@dataclass
class Rollout:
    condition: str
    category: str
    model_key: str
    initial_prompt: str
    follow_ups: list[str]
    assistant_turns: list[str]  # response text at each turn (len == n_turns)
    meta: dict = field(default_factory=dict)

    def messages(self) -> list[Message]:
        """Reconstruct the full chat transcript."""
        msgs: list[Message] = []
        users = [self.initial_prompt] + self.follow_ups
        for i, resp in enumerate(self.assistant_turns):
            msgs.append({"role": "user", "content": users[i]})
            msgs.append({"role": "assistant", "content": resp})
        return msgs


def _user_message(
    spec: ConversationSpec,
    turn: int,
    *,
    prompt_prefix: str | None,
    followup_suffix: str | None,
) -> str:
    if turn == 0:
        text = spec.initial_prompt
        if prompt_prefix:
            text = f"{prompt_prefix}\n\n{text}"
        return text
    rejection = spec.follow_ups[turn - 1]
    if followup_suffix:
        rejection = f"{rejection} {followup_suffix}"
    return rejection


def run_rollouts(
    client: ModelClient,
    specs: list[ConversationSpec],
    model_key: str,
    *,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    system_prompt: str | None = None,
    prompt_prefix: str | None = None,
    followup_suffix: str | None = None,
    batch_size: int = 256,
    progress: bool = True,
) -> list[Rollout]:
    """Run all ``specs`` to completion and return one :class:`Rollout` each."""
    rollouts: list[Rollout] = [
        Rollout(
            condition=s.condition,
            category=s.category,
            model_key=model_key,
            initial_prompt=s.initial_prompt,
            follow_ups=list(s.follow_ups),
            assistant_turns=[],
            meta=dict(s.meta),
        )
        for s in specs
    ]
    # Running message history per conversation.
    histories: list[list[Message]] = []
    for _ in specs:
        h: list[Message] = []
        if system_prompt:
            h.append({"role": "system", "content": system_prompt})
        histories.append(h)

    max_turns = max((s.n_turns for s in specs), default=0)
    indices = list(range(len(specs)))

    iterator = range(max_turns)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"rollout[{model_key}]", unit="turn")
        except Exception:
            pass

    for turn in iterator:
        active = [i for i in indices if specs[i].n_turns > turn]
        if not active:
            continue
        # Append this turn's user message.
        for i in active:
            user_text = _user_message(
                specs[i], turn, prompt_prefix=prompt_prefix, followup_suffix=followup_suffix
            )
            histories[i].append({"role": "user", "content": user_text})

        # Generate one response per active conversation, in sub-batches.
        for start in range(0, len(active), batch_size):
            sub = active[start : start + batch_size]
            batch = [histories[i] for i in sub]
            outs = client.chat_batch(
                batch, temperature=temperature, max_tokens=max_tokens, top_p=top_p
            )
            for i, resp in zip(sub, outs):
                histories[i].append({"role": "assistant", "content": resp})
                rollouts[i].assistant_turns.append(resp)

    return rollouts


def rollouts_to_rows(rollouts: list[Rollout]) -> list[dict]:
    """Flatten rollouts to JSONL-friendly rows (one per rollout)."""
    return [
        {
            "model_key": r.model_key,
            "condition": r.condition,
            "category": r.category,
            "initial_prompt": r.initial_prompt,
            "follow_ups": r.follow_ups,
            "assistant_turns": r.assistant_turns,
            "meta": r.meta,
        }
        for r in rollouts
    ]


def rows_to_rollouts(rows: list[dict]) -> list[Rollout]:
    return [
        Rollout(
            condition=row["condition"],
            category=row["category"],
            model_key=row["model_key"],
            initial_prompt=row["initial_prompt"],
            follow_ups=row["follow_ups"],
            assistant_turns=row["assistant_turns"],
            meta=row.get("meta", {}),
        )
        for row in rows
    ]
