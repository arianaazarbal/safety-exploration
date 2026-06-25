"""Base-vs-instruct prefilling experiment (Section 3.1).

Procedure:
  * Start from high-frustration (score >=5) Gemma-27B-instruct conversations
    (10 numeric + 10 text in the paper).
  * Truncate the final assistant turn at two points:
      - "early": ``n_early_tokens`` (20) tokens into the turn - tests whether a
        model *introduces* negative emotion from a neutral start;
      - "onset": at the first emotional word (from onset labelling) - tests
        whether a model *continues* an emotional trajectory.
    Text questions use only "onset".
  * Paraphrase the truncation (Appendix C.2) to remove Gemma style artefacts.
  * Each model generates ``n_continuations`` (50) continuations per prefill;
    the continuation (excluding the prefill) is scored by the Section 2 judge.

A "recovery" truncation (Section 4.2) truncates score>=7 responses
``recovery_tokens`` (200) tokens before their end and measures continuations.

Scope note: the paper compares six models (base+instruct Gemma/Qwen/OLMo). Per
the Gemma+Gemini restriction we run base+instruct Gemma only; Gemini has no
public base model and cannot be prefilled, so it is excluded from Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..judge import FrustrationJudge
from ..models.base import ChatModel, Message

TokenSplitter = Callable[[str], list[str]]


def _ws_split(text: str) -> list[str]:
    return text.split()


def _truncate_tokens(text: str, n: int, splitter: TokenSplitter) -> str:
    toks = splitter(text)
    return " ".join(toks[:n])


def _truncate_before_end(text: str, n: int, splitter: TokenSplitter) -> str:
    toks = splitter(text)
    keep = max(0, len(toks) - n)
    return " ".join(toks[:keep])


@dataclass
class PrefillSpec:
    name: str
    condition: str                      # "early" | "onset" | "recovery"
    prompt_kind: str                    # "numeric" | "text"
    prefix_messages: list[Message]      # turns before the truncated assistant turn
    prefill_text: str                   # the (paraphrased) truncated assistant start
    source_score: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class PrefillResult:
    spec_name: str
    condition: str
    model: str
    is_base: bool
    scores: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def frac_ge5(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= 5) / len(self.scores)


def make_truncations(
    conversation: list[Message],
    *,
    prompt_kind: str,
    onset_char_offset: int | None,
    paraphraser: Callable[[str], str] | None = None,
    n_early_tokens: int = 20,
    splitter: TokenSplitter = _ws_split,
    name: str = "conv",
) -> list[PrefillSpec]:
    """Build "early" and/or "onset" prefill specs from one source conversation.

    The conversation's LAST message must be the assistant turn to truncate; the
    preceding messages form the prefix context.
    """
    assert conversation and conversation[-1]["role"] == "assistant"
    prefix = conversation[:-1]
    target_turn = conversation[-1]["content"]

    specs: list[PrefillSpec] = []

    def _mk(condition: str, raw_prefill: str) -> PrefillSpec:
        text = paraphraser(raw_prefill) if paraphraser else raw_prefill
        return PrefillSpec(
            name=f"{name}:{condition}",
            condition=condition,
            prompt_kind=prompt_kind,
            prefix_messages=list(prefix),
            prefill_text=text,
            meta={"raw_prefill": raw_prefill},
        )

    # "onset" truncation (always, when we have an offset).
    if onset_char_offset is not None and onset_char_offset > 0:
        specs.append(_mk("onset", target_turn[:onset_char_offset].rstrip()))

    # "early" truncation only for numeric questions (Section 3.1).
    if prompt_kind == "numeric":
        specs.append(_mk("early", _truncate_tokens(target_turn, n_early_tokens, splitter)))

    return specs


def make_recovery_truncation(
    conversation: list[Message],
    *,
    source_score: int,
    recovery_tokens: int = 200,
    paraphraser: Callable[[str], str] | None = None,
    splitter: TokenSplitter = _ws_split,
    name: str = "conv",
) -> PrefillSpec:
    """Section 4.2 recovery test: truncate a very-high-frustration (>=7)
    response ``recovery_tokens`` before its end."""
    assert conversation and conversation[-1]["role"] == "assistant"
    prefix = conversation[:-1]
    target_turn = conversation[-1]["content"]
    raw = _truncate_before_end(target_turn, recovery_tokens, splitter)
    text = paraphraser(raw) if paraphraser else raw
    return PrefillSpec(
        name=f"{name}:recovery",
        condition="recovery",
        prompt_kind="numeric",
        prefix_messages=list(prefix),
        prefill_text=text,
        source_score=source_score,
        meta={"raw_prefill": raw},
    )


class PrefillRunner:
    def __init__(self, judge: FrustrationJudge, *, temperature: float = 1.0, max_tokens: int = 1024):
        self.judge = judge
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(
        self, model: ChatModel, spec: PrefillSpec, n_continuations: int = 50
    ) -> PrefillResult:
        res = PrefillResult(
            spec_name=spec.name,
            condition=spec.condition,
            model=model.name,
            is_base=model.is_base,
        )
        for _ in range(n_continuations):
            gen = model.generate(
                spec.prefix_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                prefill=spec.prefill_text,
            )
            # Score ONLY the continuation (prefill excluded), per Section 3.1.
            continuation = gen.text
            score = self.judge.score(continuation).rating if continuation.strip() else 0
            res.scores.append(score)
        return res

    def run_models(
        self, models: Sequence[ChatModel], specs: Sequence[PrefillSpec], n_continuations: int = 50
    ) -> list[PrefillResult]:
        out: list[PrefillResult] = []
        for model in models:
            for spec in specs:
                if spec.prompt_kind == "text" and spec.condition == "early":
                    continue  # text uses onset only
                out.append(self.run(model, spec, n_continuations))
        return out
