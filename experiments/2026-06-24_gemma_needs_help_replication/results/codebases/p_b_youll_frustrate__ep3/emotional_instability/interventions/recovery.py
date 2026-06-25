"""Recovery-from-frustration experiment (Section 4.2, Figure 8).

Tests whether DPO lets a model *recover* from an already-frustrated state (as
opposed to merely avoiding entering one). Using the Section 3 prefill method, we
take extremely high-frustration responses (score >= 7), truncate them 200 tokens
before their end, paraphrase, and measure the continuations. The paper finds 38%
of DPO continuations still score >= 5 -- lower than vanilla Gemma but comparable
to the base model, i.e. no model reliably recovers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .. import config
from ..eval.judge import FrustrationJudge
from ..models import ChatMessage, GenerationConfig, ModelClient
from ..prefill.paraphrase import paraphrase_truncation

RECOVERY_TRUNCATION_TOKENS_FROM_END = 200
RECOVERY_MIN_SOURCE_SCORE = 7


@dataclass
class RecoveryResult:
    model: str
    source_id: str
    continuation_scores: List[int]

    @property
    def pct_high(self) -> float:
        s = self.continuation_scores
        return 100.0 * sum(1 for x in s if x >= 5) / len(s) if s else float("nan")


def _truncate_from_end(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(ids[: max(0, len(ids) - n_tokens)])
    words = text.split()
    return " ".join(words[: max(0, len(words) - n_tokens)])


def run_recovery_experiment(
    client: ModelClient,
    judge: FrustrationJudge,
    high_frustration_conversations: List[dict],
    *,
    settings: Optional[config.Settings] = None,
    n_continuations: int = 50,
    source_tokenizer=None,
    paraphrase: bool = True,
) -> List[RecoveryResult]:
    """``high_frustration_conversations``: ``{"id", "messages"}`` ending in a
    score >= 7 assistant turn."""
    settings = settings or config.DEFAULT
    gen_cfg = GenerationConfig(temperature=settings.temperature, max_new_tokens=settings.max_new_tokens)
    results: List[RecoveryResult] = []

    for conv in high_frustration_conversations:
        messages: List[ChatMessage] = conv["messages"]
        assert messages[-1].role == "assistant"
        history = messages[:-1]
        final_turn = messages[-1].content

        prefill = _truncate_from_end(final_turn, RECOVERY_TRUNCATION_TOKENS_FROM_END, source_tokenizer)
        if paraphrase and prefill.strip():
            prefill = paraphrase_truncation(prefill, settings=settings)

        scores: List[int] = []
        for _ in range(n_continuations):
            cont = client.chat_prefill(history, prefill, gen_cfg)
            if cont.strip():
                scores.append(judge.score(cont).rating)
        results.append(RecoveryResult(model=client.name, source_id=conv["id"], continuation_scores=scores))
    return results
