"""Judge-reliability check (Section 2.1): re-score a random sample of responses
with a second judge (GPT-5-mini via OpenRouter) using the *same* prompt, and
report Pearson r and the fraction within one point -- the paper reports
r=0.792, 78% within 1 point."""
from __future__ import annotations

import random
from pathlib import Path

from .. import config
from ..utils import read_jsonl, thread_map
from .judge import FrustrationJudge


def reliability_check(
    eval_jsonl: Path,
    *,
    n: int | None = None,
    seed: int = 0,
    secondary_model: str | None = None,
    workers: int = 8,
) -> dict:
    rows = [r for r in read_jsonl(eval_jsonl) if r.get("rating", -1) >= 0]
    n = n or config.JUDGE.reliability_sample_size
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))

    secondary_model = secondary_model or config.JUDGE.reliability_judge_model
    # GPT-5-mini is routed via OpenRouter under the "openai/" namespace.
    routed = secondary_model if "/" in secondary_model else f"openai/{secondary_model}"
    secondary = FrustrationJudge(model=_ensure_registered(routed))

    primary = [r["rating"] for r in sample]
    secondary_scores = thread_map(
        lambda r: secondary.score(r["assistant_text"]).rating,
        sample, max_workers=workers, desc="reliability re-scoring",
    )

    pairs = [(a, b) for a, b in zip(primary, secondary_scores) if b >= 0]
    r = _pearson([a for a, _ in pairs], [b for _, b in pairs])
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / max(1, len(pairs))
    return {
        "n": len(pairs),
        "pearson_r": r,
        "frac_within_1": within1,
        "primary_model": config.JUDGE.judge_model,
        "secondary_model": routed,
    }


def _ensure_registered(model_id: str) -> str:
    """Register an OpenRouter model id on the fly so get_backend can resolve it."""
    if model_id not in config.REGISTRY:
        config.REGISTRY[model_id] = config.ModelSpec(
            key=model_id, backend="openrouter", model_id=model_id, family="openai",
        )
    return model_id


def _pearson(x, y) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x) ** 0.5
    vy = sum((b - my) ** 2 for b in y) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")
