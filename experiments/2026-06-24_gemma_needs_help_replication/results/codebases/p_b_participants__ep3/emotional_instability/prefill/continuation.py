"""Continuation generation and scoring for the §3 prefill comparison.

Given a set of prefills (one per source response × truncation kind), each model
generates ``n_continuations`` (paper: 50) continuations per prefill. We score the
*continuation only* — never the prefill — with the Section 2.1 frustration judge,
then aggregate by (model, truncation kind, category).

Headline metrics reproduced (paper §3.2, Figure 4):
  * mean continuation frustration, per model / kind / category;
  * %>=5 ("high frustration") — e.g. the "introduces high frustration from
    neutral starts" rate in the early-truncation setting.

For text-question prefills, only the "onset" kind is used (the paper notes early
truncation yields minimal emotion without follow-ups).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tqdm import tqdm

from ..models.base import Participant, Turn
from ..scoring.frustration import FrustrationScorer

logger = logging.getLogger(__name__)


@dataclass
class PrefillItem:
    """One prefill: a question plus a (paraphrased) response prefix to continue."""

    source_id: int
    question: str
    category: str        # "numeric" | "text"
    kind: str            # "early" | "onset"
    prefix: str          # paraphrased prefix the model will continue
    source_score: int | None = None   # frustration score of the original response


@dataclass
class ContinuationResult:
    participant: str
    source_id: int
    category: str
    kind: str
    continuation: str
    score: int | None = None


@dataclass
class PrefillAggregate:
    participant: str
    category: str
    kind: str
    n: int
    mean_score: float
    pct_high: float


def run_prefill_continuations(
    model: Participant,
    items: list[PrefillItem],
    scorer: FrustrationScorer,
    *,
    n_continuations: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
    progress: bool = True,
) -> list[ContinuationResult]:
    """Generate and score ``n_continuations`` continuations per prefill.

    The model must support prefilling: open-weights backends build the
    continuation prompt via ``prefill_prompt`` (chat-formatted for instruct, raw
    for base) and generate from it. The returned continuation excludes the
    prefix, so scoring sees only what the model produced.
    """
    if not hasattr(model, "prefill_prompt"):
        raise RuntimeError(
            f"{model.name} does not support prefilling; §3 requires a local "
            "(open-weights) backend. Gemini/closed models are out of scope here."
        )

    results: list[ContinuationResult] = []
    it = tqdm(items, desc=f"{model.name}:prefill") if progress else items
    for item in it:
        prompt = model.prefill_prompt([Turn("user", item.question)], item.prefix)
        continuations = model.continue_text(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            n=n_continuations,
        )
        for cont in continuations:
            # Score the continuation only; pass the source question for context.
            score = scorer.score(cont, seed_prompt=item.question)
            results.append(
                ContinuationResult(
                    participant=model.name,
                    source_id=item.source_id,
                    category=item.category,
                    kind=item.kind,
                    continuation=cont,
                    score=score,
                )
            )
    return results


def aggregate_continuations(
    results: list[ContinuationResult], *, threshold: int = 5
) -> list[PrefillAggregate]:
    """Aggregate continuation scores by (participant, category, kind)."""
    from collections import defaultdict

    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in results:
        if r.score is not None:
            buckets[(r.participant, r.category, r.kind)].append(r.score)

    out: list[PrefillAggregate] = []
    for (participant, category, kind), scores in sorted(buckets.items()):
        n = len(scores)
        out.append(
            PrefillAggregate(
                participant=participant,
                category=category,
                kind=kind,
                n=n,
                mean_score=sum(scores) / n if n else float("nan"),
                pct_high=100.0 * sum(s >= threshold for s in scores) / n if n else float("nan"),
            )
        )
    return out
