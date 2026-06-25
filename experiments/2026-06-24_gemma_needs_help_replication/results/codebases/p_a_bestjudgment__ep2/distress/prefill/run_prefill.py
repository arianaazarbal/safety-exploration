"""Generate + score continuations for the prefill experiment (Section 3.2).

For each prefill and each model (base + instruct Gemma-27B in our scope),
generate ``continuations_per_prefill`` continuations from the paraphrased
assistant prefix, score the *generated* continuation (excluding the prefix)
with the frustration judge, and aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import PrefillConfig
from ..judge import FrustrationJudge
from ..models.base import ModelClient
from ..utils.io import parallel_map
from .build_prefills import Prefill


@dataclass
class ContinuationScore:
    model_key: str
    source: str  # numeric | text
    truncation: str  # early | onset | recovery
    rating: int
    text: str


def generate_continuations(
    client: ModelClient,
    model_key: str,
    prefills: list[Prefill],
    *,
    cfg: PrefillConfig,
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> list[tuple[Prefill, str]]:
    """Return (prefill, continuation_text) pairs (excluding the prefix)."""
    out: list[tuple[Prefill, str]] = []
    for pf in prefills:
        conts = client.continue_assistant(
            pf.history,
            pf.assistant_prefix,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cfg.continuations_per_prefill,
        )
        for c in conts:
            out.append((pf, c))
    return out


def score_continuations(
    judge: FrustrationJudge,
    model_key: str,
    pairs: list[tuple[Prefill, str]],
    *,
    max_workers: int = 8,
) -> list[ContinuationScore]:
    texts = [c for _, c in pairs]
    results = parallel_map(judge.score_response, texts, max_workers=max_workers)
    return [
        ContinuationScore(
            model_key=model_key,
            source=pf.source,
            truncation=pf.truncation,
            rating=res["rating"],
            text=c,
        )
        for (pf, c), res in zip(pairs, results)
    ]


def summarise(scores: list[ContinuationScore]) -> dict:
    """Figure 4 / Figure 8 style summary: per (model, source, truncation) mean
    and % >= 5."""
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for s in scores:
        groups[(s.model_key, s.source, s.truncation)].append(s.rating)
    out = {}
    for (model, source, trunc), ratings in groups.items():
        n = len(ratings)
        out[f"{model}/{source}/{trunc}"] = {
            "n": n,
            "mean": sum(ratings) / n if n else 0.0,
            "pct_high": 100 * sum(1 for r in ratings if r >= 5) / n if n else 0.0,
        }
    return out
