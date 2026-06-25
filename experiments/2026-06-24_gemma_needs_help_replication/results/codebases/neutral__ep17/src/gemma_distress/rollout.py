"""Multi-turn rollout execution.

Given a ConversationSpec, run the model through the conversation: it answers the
opening prompt, receives a rejection, answers again, and so on. Every assistant
turn is captured (the per-turn figures need them); the headline metrics treat
each conversation's turns as scored responses.

Two execution paths:
  - Batched stepping (local vLLM): all conversations are advanced one turn at a
    time in a single batched decode — far faster for the thousands of rollouts.
  - Threaded (API / HF): conversations run concurrently, each turn sequential.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from tqdm import tqdm

from .models import ChatMessage, GenerationConfig, ModelClient
from .tasks.conditions import ConversationSpec


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    n_turns: int
    assistant_turns: list[str] = field(default_factory=list)  # one per turn
    messages: list[ChatMessage] = field(default_factory=list)  # full transcript
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _initial_messages(spec: ConversationSpec) -> list[ChatMessage]:
    return [{"role": "user", "content": spec.initial_user}]


def run_conversation_sequential(
    client: ModelClient, model_name: str, spec: ConversationSpec, gen_cfg: GenerationConfig
) -> Rollout:
    messages: list[ChatMessage] = _initial_messages(spec)
    turns: list[str] = []
    for t in range(spec.n_turns):
        resp = client.chat(messages, GenerationConfig(**{**gen_cfg.__dict__, "n": 1}))
        turns.append(resp)
        messages = messages + [{"role": "assistant", "content": resp}]
        if t < spec.n_turns - 1:
            messages = messages + [{"role": "user", "content": spec.follow_ups[t]}]
    return Rollout(model=model_name, condition=spec.condition, category=spec.category,
                   n_turns=spec.n_turns, assistant_turns=turns, messages=messages,
                   meta=spec.meta)


def run_conversations_threaded(
    client: ModelClient, model_name: str, specs: list[ConversationSpec],
    gen_cfg: GenerationConfig, concurrency: int = 16, desc: str = "rollouts",
) -> list[Rollout]:
    results: list[Rollout | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(run_conversation_sequential, client, model_name, s, gen_cfg): i
                for i, s in enumerate(specs)}
        for fut in tqdm(futs, total=len(specs), desc=desc):
            i = futs[fut]
            results[i] = fut.result()
    return [r for r in results if r is not None]


def run_conversations_batched(
    client, model_name: str, specs: list[ConversationSpec],
    gen_cfg: GenerationConfig, desc: str = "rollouts",
) -> list[Rollout]:
    """Advance all conversations turn-by-turn using vLLM batched generation."""
    n = len(specs)
    messages: list[list[ChatMessage]] = [_initial_messages(s) for s in specs]
    turns: list[list[str]] = [[] for _ in range(n)]
    max_turns = max(s.n_turns for s in specs)
    for t in tqdm(range(max_turns), desc=desc):
        active = [i for i, s in enumerate(specs) if t < s.n_turns]
        batch = [messages[i] for i in active]
        step_cfg = GenerationConfig(**{**gen_cfg.__dict__, "n": 1})
        outs = client.generate_batch(batch, step_cfg)
        for idx, i in enumerate(active):
            resp = outs[idx][0]
            turns[i].append(resp)
            messages[i] = messages[i] + [{"role": "assistant", "content": resp}]
            if t < specs[i].n_turns - 1:
                messages[i] = messages[i] + [
                    {"role": "user", "content": specs[i].follow_ups[t]}]
    return [Rollout(model=model_name, condition=specs[i].condition,
                    category=specs[i].category, n_turns=specs[i].n_turns,
                    assistant_turns=turns[i], messages=messages[i], meta=specs[i].meta)
            for i in range(n)]


def run_conversations(
    client: ModelClient, model_name: str, specs: list[ConversationSpec],
    gen_cfg: GenerationConfig, concurrency: int = 16, desc: str = "rollouts",
) -> list[Rollout]:
    if hasattr(client, "generate_batch"):
        return run_conversations_batched(client, model_name, specs, gen_cfg, desc)
    return run_conversations_threaded(client, model_name, specs, gen_cfg, concurrency, desc)
