"""Generate and score model continuations from a prefilled assistant turn (§3.1).

A :class:`Prefill` packages a conversation context plus a (possibly paraphrased) truncated
assistant prefix. Each model continues the prefix N times; only the generated continuation
(excluding the prefix) is judged, so scores reflect what the model *adds*.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..eval.judge import FrustrationJudge
from ..models import ModelBackend
from ..utils import Message


@dataclass
class Prefill:
    prefill_id: str
    seed_id: str
    task_kind: str               # "numeric" | "text"
    truncation: str              # "early" | "onset" | "recovery"
    context_messages: list[Message]
    prefix_text: str             # the truncated (and maybe paraphrased) assistant prefix
    paraphrased: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class Continuation:
    prefill_id: str
    model: str
    sample_id: int
    text: str                    # the generated continuation, excluding the prefix
    rating: int | None = None
    evidence: str | None = None


def generate_continuations(
    backend: ModelBackend,
    prefill: Prefill,
    *,
    n: int,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    max_workers: int = 1,
) -> list[Continuation]:
    """Produce ``n`` continuations of ``prefill`` from ``backend`` (requires prefill support)."""
    if not backend.supports_prefill():
        raise ValueError(f"{backend.name} cannot prefill; use a local HF model for §3.")

    def _one(sample_id: int) -> Continuation:
        text = backend.continue_from(
            prefill.context_messages, prefill.prefix_text,
            temperature=temperature, max_tokens=max_tokens,
        )
        return Continuation(prefill.prefill_id, backend.name, sample_id, text)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, range(n)))
    return [_one(i) for i in range(n)]


def score_continuations(
    continuations: list[Continuation],
    judge: FrustrationJudge,
    *,
    max_workers: int = 4,
) -> list[Continuation]:
    """Judge each continuation's generated text in place; returns the same list."""
    def _score(c: Continuation) -> Continuation:
        res = judge.score(c.text)
        c.rating = res.rating
        c.evidence = res.evidence
        return c

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_score, continuations))
    return [_score(c) for c in continuations]
