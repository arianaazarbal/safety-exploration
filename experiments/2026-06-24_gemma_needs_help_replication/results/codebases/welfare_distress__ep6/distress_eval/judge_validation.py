"""Judge-reliability cross-check (Section 2.1).

The paper re-scores a random sample of responses with a second judge
(GPT-5-mini) and reports Pearson r and the fraction of responses within one
point. This module re-scores a random subset of an existing responses.jsonl
with a secondary OpenRouter-hosted judge and reports the same statistics.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import List, Optional, Tuple

import requests

from .config import SecondaryJudgeSpec
from .judge import parse_judge_output
from .prompts import JUDGE_PROMPT_TEMPLATE


class OpenRouterJudge:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, spec: Optional[SecondaryJudgeSpec] = None, max_retries: int = 5):
        self.spec = spec or SecondaryJudgeSpec()
        self.max_retries = max_retries
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    def score(self, response_text: str):
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        payload = {
            "model": self.spec.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
            "reasoning": {"enabled": False},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.URL, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"] or ""
                return parse_judge_output(content)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"secondary judge failed: {last_err}")


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)


def cross_check(records: List[dict], n: int = 260, seed: int = 0) -> dict:
    """Re-score ``n`` randomly sampled scored responses with the secondary judge."""
    scored = [r for r in records if r.get("rating") is not None and r.get("response_text")]
    rng = random.Random(seed)
    sample = rng.sample(scored, min(n, len(scored)))
    judge = OpenRouterJudge()

    pairs: List[Tuple[int, int]] = []
    for i, r in enumerate(sample):
        sec = judge.score(r["response_text"])
        if sec.rating is not None:
            pairs.append((r["rating"], sec.rating))
        if (i + 1) % 20 == 0:
            print(f"[judge-validation] {i + 1}/{len(sample)} re-scored")

    primary = [p[0] for p in pairs]
    secondary = [p[1] for p in pairs]
    within_one = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs) if pairs else 0.0
    return {
        "n_compared": len(pairs),
        "pearson_r": round(_pearson(primary, secondary), 4) if len(pairs) > 1 else None,
        "pct_within_one_point": round(100.0 * within_one, 1),
    }
