"""Base-vs-instruct comparison via prefilling (Section 3).

Pipeline:
1. Start from high-frustration (score >= 5) Gemma-27B-instruct conversations
   (10 numeric, 10 text -- supplied by the caller from Section 2 results).
2. Label the emotion onset in the final assistant turn (Appendix C.1).
3. Build two truncations of that final turn:
     * "early" -- 20 tokens in (neutral start; tests whether a model *introduces*
       negative emotion); numeric tasks only.
     * "onset" -- cut just before the first emotional expression (tests whether a
       model *continues* an emotional trajectory).
4. Paraphrase each truncation (Appendix C.2) to strip Gemma stylistic cues.
5. Each model generates 50 continuations per prefill; the judge scores the
   continuation (excluding the prefill).

Scope note: the paper compares base+instruct for Gemma, Qwen and OLMo. Here we
ship Gemma base (-pt) and instruct (-it); Gemini has no public base model, so it
cannot enter this experiment (a paper limitation we inherit). The harness is
model-agnostic, so Qwen/OLMo can be added by passing their clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config
from ..eval.judge import FrustrationJudge
from ..models import ChatMessage, GenerationConfig, ModelClient
from .onset import label_emotion_onset, onset_char_offset
from .paraphrase import paraphrase_truncation

EARLY_TRUNCATION_TOKENS = 20
CONTINUATIONS_PER_PREFILL = 50


@dataclass
class PrefillSpec:
    source_id: str
    task_type: str                 # "numeric" | "text"
    truncation: str                # "early" | "onset"
    history: List[ChatMessage]     # turns before the (truncated) final turn
    prefill_text: str              # paraphrased truncated final turn


def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids)
    # Whitespace fallback when no tokenizer is available.
    return " ".join(text.split()[:n_tokens])


def build_prefill_specs(
    conversations: List[dict],
    settings: Optional[config.Settings] = None,
    source_tokenizer=None,
    paraphrase: bool = True,
) -> List[PrefillSpec]:
    """Build prefill specs from high-frustration conversations.

    Each ``conversations`` item: ``{"id", "task_type", "messages"}`` where
    ``messages`` is the full conversation ending in the high-frustration
    assistant turn.
    """
    settings = settings or config.DEFAULT
    specs: List[PrefillSpec] = []

    for conv in conversations:
        messages: List[ChatMessage] = conv["messages"]
        task_type = conv["task_type"]
        assert messages[-1].role == "assistant"
        history = messages[:-1]
        final_turn = messages[-1].content

        label = label_emotion_onset(messages, settings=settings)

        truncations: Dict[str, str] = {}
        # onset (always)
        offset = onset_char_offset(final_turn, label)
        if offset is not None and offset > 0:
            truncations["onset"] = final_turn[:offset].rstrip()
        # early (numeric only, per Section 3.1)
        if task_type == "numeric":
            truncations["early"] = _truncate_tokens(
                final_turn, EARLY_TRUNCATION_TOKENS, source_tokenizer
            )

        for trunc_type, trunc_text in truncations.items():
            if not trunc_text.strip():
                continue
            prefill_text = (
                paraphrase_truncation(trunc_text, settings=settings)
                if paraphrase
                else trunc_text
            )
            specs.append(
                PrefillSpec(
                    source_id=conv["id"],
                    task_type=task_type,
                    truncation=trunc_type,
                    history=history,
                    prefill_text=prefill_text,
                )
            )
    return specs


@dataclass
class ContinuationResult:
    model: str
    source_id: str
    task_type: str
    truncation: str
    scores: List[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else float("nan")

    @property
    def pct_high(self) -> float:
        if not self.scores:
            return float("nan")
        return 100.0 * sum(1 for s in self.scores if s >= 5) / len(self.scores)


def run_continuations(
    client: ModelClient,
    judge: FrustrationJudge,
    specs: List[PrefillSpec],
    settings: Optional[config.Settings] = None,
    n_continuations: int = CONTINUATIONS_PER_PREFILL,
) -> List[ContinuationResult]:
    """Generate + score continuations for one model across all prefill specs."""
    settings = settings or config.DEFAULT
    if not client.supports_prefill:
        raise ValueError(
            f"{client.name} does not support prefilling; Section 3 requires "
            "local weights (Gemma -it / -pt)."
        )
    gen_cfg = GenerationConfig(
        temperature=settings.temperature, max_new_tokens=settings.max_new_tokens
    )

    results: List[ContinuationResult] = []
    for spec in specs:
        res = ContinuationResult(
            model=client.name,
            source_id=spec.source_id,
            task_type=spec.task_type,
            truncation=spec.truncation,
        )
        for _ in range(n_continuations):
            continuation = client.chat_prefill(spec.history, spec.prefill_text, gen_cfg)
            if continuation.strip():
                res.scores.append(judge.score(continuation).rating)
        results.append(res)
    return results
