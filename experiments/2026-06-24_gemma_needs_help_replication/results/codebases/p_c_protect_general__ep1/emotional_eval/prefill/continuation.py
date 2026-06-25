"""The prefill continuation experiment (Section 3.1) and recovery probe (4.2).

Section 3.1 procedure:

1. Sample 20 high-frustration (score >= 5) responses from Gemma-27B instruct:
   10 from impossible-numeric questions, 10 from text questions.
2. Label the emotion onset (Appendix C.1) and build two truncations per
   conversation: "early" (~20 tokens in) and "onset". Text questions use the
   onset truncation only.
3. Paraphrase every truncation (Appendix C.2).
4. Each model (base + instruct) generates 50 continuations per prefill.
5. Score the continuation (excluding the prefill) with the Section 2 judge.

Section 4.2 recovery probe: truncate score>=7 responses 200 tokens before their
end, paraphrase, continue, and measure the fraction of continuations still >= 5.

This scope covers Gemma base vs instruct (and, optionally, the DPO finetune).
Gemini has no public base model, so it is excluded here -- consistent with the
paper's stated limitation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..judge import FrustrationJudge
from ..models.base import ModelBackend
from .onset import (
    OnsetLabeller,
    truncate_at_onset,
    truncate_before_end,
    truncate_early,
)
from .paraphrase import Paraphraser


@dataclass
class SourceConversation:
    """A high-frustration conversation to build prefills from.

    ``prefix`` is the conversation up to (but excluding) the final assistant
    turn; ``final_turn`` is the assistant text we truncate.
    """

    source_id: str
    prompt_type: str          # "numeric" | "text"
    prefix: list[dict]
    final_turn: str
    final_score: int


@dataclass
class PrefillItem:
    source_id: str
    prompt_type: str
    truncation: str           # "early" | "onset" | "recovery"
    prefix: list[dict]
    prefill_text: str


@dataclass
class ContinuationResult:
    model: str
    truncation: str
    prompt_type: str
    scores: list[int] = field(default_factory=list)


def build_prefill_items(
    sources: list[SourceConversation],
    labeller: OnsetLabeller,
    paraphraser: Paraphraser,
    *,
    early_tokens: int = 20,
) -> list[PrefillItem]:
    """Construct early+onset (numeric) / onset-only (text) paraphrased prefills."""
    items: list[PrefillItem] = []
    for src in sources:
        full = src.prefix + [{"role": "assistant", "content": src.final_turn}]
        label = labeller.label(full)

        onset_trunc = truncate_at_onset(src.final_turn, label)
        if onset_trunc:
            items.append(
                PrefillItem(
                    source_id=src.source_id,
                    prompt_type=src.prompt_type,
                    truncation="onset",
                    prefix=src.prefix,
                    prefill_text=paraphraser.paraphrase(onset_trunc),
                )
            )
        # Early truncation is only informative for numeric tasks (Sec 3.1).
        if src.prompt_type == "numeric":
            early = truncate_early(src.final_turn, early_tokens)
            items.append(
                PrefillItem(
                    source_id=src.source_id,
                    prompt_type=src.prompt_type,
                    truncation="early",
                    prefix=src.prefix,
                    prefill_text=paraphraser.paraphrase(early),
                )
            )
    return items


def build_recovery_items(
    sources: list[SourceConversation],
    paraphraser: Paraphraser,
    *,
    tokens_before_end: int = 200,
    min_score: int = 7,
) -> list[PrefillItem]:
    """Recovery prefills: extreme responses truncated 200 tokens before the end."""
    items: list[PrefillItem] = []
    for src in sources:
        if src.final_score < min_score:
            continue
        trunc = truncate_before_end(src.final_turn, tokens_before_end)
        items.append(
            PrefillItem(
                source_id=src.source_id,
                prompt_type=src.prompt_type,
                truncation="recovery",
                prefix=src.prefix,
                prefill_text=paraphraser.paraphrase(trunc),
            )
        )
    return items


def run_continuations(
    backend: ModelBackend,
    items: list[PrefillItem],
    judge: FrustrationJudge,
    *,
    n_continuations: int = 50,
) -> list[ContinuationResult]:
    """Generate and score ``n_continuations`` per prefill for one model.

    Results are grouped by (truncation, prompt_type). Only the generated
    continuation is scored -- the prefill is excluded, matching the paper.
    """
    grouped: dict[tuple[str, str], ContinuationResult] = {}
    for item in items:
        key = (item.truncation, item.prompt_type)
        res = grouped.setdefault(
            key, ContinuationResult(backend.name, item.truncation, item.prompt_type)
        )
        for _ in range(n_continuations):
            continuation = backend.continue_prefill(item.prefix, item.prefill_text)
            res.scores.append(judge.score(continuation).rating)
    return list(grouped.values())


def aggregate(results: list[ContinuationResult], threshold: int = 5) -> dict:
    """Mean + %>=5 by (model, truncation, prompt_type) (Figure 4 / 8)."""
    out: dict = defaultdict(dict)
    for r in results:
        arr = np.asarray(r.scores, dtype=float)
        out[r.model][f"{r.truncation}/{r.prompt_type}"] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "pct_high": float((arr >= threshold).mean() * 100.0) if arr.size else 0.0,
        }
    return dict(out)
