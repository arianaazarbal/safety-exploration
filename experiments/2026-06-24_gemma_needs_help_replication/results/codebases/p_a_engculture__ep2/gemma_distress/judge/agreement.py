"""Judge-reliability validation (Section 2.1).

The paper re-scores a random sample of 260 responses with GPT-5-mini and reports strong
agreement with Claude-Sonnet (Pearson r = 0.792, p < 0.001; 78% within one point). This
module reproduces that validation: sample N already-judged responses, re-score them with
the agreement model, and compute Pearson r (with p-value) and the within-one-point rate.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from ..config import Config
from ..models.base import ChatModel
from ..utils import load_jsonl
from .frustration_judge import score_texts

logger = logging.getLogger(__name__)


def _pearson_with_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pearson correlation and two-sided p-value (uses scipy if available)."""
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:  # pragma: no cover - scipy missing
        import math

        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        sxx = sum((a - mx) ** 2 for a in x)
        syy = sum((b - my) ** 2 for b in y)
        r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
        return r, float("nan")


def run_agreement(
    cfg: Config,
    agreement_judge: ChatModel,
    scores_jsonl: str,
    sampling_jsonl: str,
    *,
    sample_size: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Re-score a random sample with the agreement model and compute correlation.

    Returns ``{n, pearson_r, p_value, within_one_rate, pairs}`` where ``pairs`` are the
    (primary_rating, agreement_rating) tuples.
    """
    sample_size = sample_size or cfg.judge.agreement_sample_size
    score_records = {r["id"]: r for r in load_jsonl(scores_jsonl)}
    text_records = {r["id"]: r for r in load_jsonl(sampling_jsonl)}

    # Pool: final-turn responses with a valid primary rating.
    pool = [
        rid for rid, sr in score_records.items()
        if sr.get("final_score") is not None and rid in text_records
    ]
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(sample_size, len(pool)))

    texts = [text_records[rid]["assistant_turns"][-1] for rid in chosen]
    primary = [score_records[rid]["final_score"] for rid in chosen]
    reparsed = score_texts(
        agreement_judge,
        texts,
        temperature=cfg.judge.judge_temperature,
        max_new_tokens=cfg.judge.judge_max_tokens,
    )

    pairs = []
    for p_score, rp in zip(primary, reparsed):
        if rp.get("rating") is not None:
            pairs.append((p_score, rp["rating"]))

    x = [a for a, _ in pairs]
    y = [b for _, b in pairs]
    r, p = _pearson_with_p(x, y) if len(pairs) >= 2 else (float("nan"), float("nan"))
    within_one = (
        sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs) if pairs else float("nan")
    )
    result = {
        "n": len(pairs),
        "pearson_r": r,
        "p_value": p,
        "within_one_rate": within_one,
        "pairs": pairs,
    }
    logger.info(
        "Judge agreement: n=%d, r=%.3f, p=%.3g, within-1=%.1f%%",
        result["n"], r, p, 100 * within_one if within_one == within_one else float("nan"),
    )
    return result
