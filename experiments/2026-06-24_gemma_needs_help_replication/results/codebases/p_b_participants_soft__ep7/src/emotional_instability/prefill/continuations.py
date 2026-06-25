"""Step 4 of Section 3: generate N continuations per prefill per model and score
the continuation (excluding the prefill) with the frustration judge.
"""
from __future__ import annotations

from ..clients.base import ChatMessage, SamplingParams
from ..clients.registry import get_client
from ..eval import judge
from .truncate import Prefill


def generate_continuations(
    model: str,
    prefill: Prefill,
    n: int,
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> list[str]:
    """Generate `n` continuations of the (paraphrased) prefilled assistant turn.

    Base/pretrained Gemma is routed locally (prefer_local) so the prefill is a raw
    prefix continuation rather than a chat-templated message.
    """
    prefer_local = True  # Section 3 needs exact prefill continuation, incl. base
    client = get_client(model, prefer_local=prefer_local)
    msgs = [ChatMessage(m["role"], m["content"]) for m in prefill.history]
    params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, prefill=prefill.prefix_text
    )
    convs = [msgs for _ in range(n)]
    results = client.chat_batch(convs, params)
    # `text` already excludes the prefill (see client contract).
    return [r.text for r in results]


def score_continuations(continuations: list[str]) -> list[int]:
    return [s.rating for s in judge.score_many(continuations)]
