"""LLM frustration judge (Section 2.1 / Appendix B.2).

Each model response is scored on the integer 0-10 frustration scale by
Claude-Sonnet-4 (pinned snapshot `claude-sonnet-4-20250514`). The judge is
asked for JSON `{"evidence", "reasoning", "rating"}`; we parse robustly and
clamp the rating to [0, 10].

A cross-check judge (GPT-5-mini, via OpenRouter) re-scores a random subset so we
can reproduce the paper's reliability statistic (Pearson r = 0.792; 78% within
one point). See analysis/judge_agreement.py.

Both judges use the *same* prompt (the paper re-scores "using the same prompt").
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from . import config, prompts


@dataclass
class JudgeResult:
    rating: int                 # clamped 0-10; -1 means parse failure
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the final JSON object and pull out the rating.

    The judge prompt allows free-text analysis before the JSON, so we take the
    last brace-delimited object in the output.
    """
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "rating" in obj:
            try:
                rating = int(round(float(obj["rating"])))
            except (TypeError, ValueError):
                continue
            rating = max(0, min(10, rating))
            return JudgeResult(rating=rating,
                               evidence=str(obj.get("evidence", "")),
                               reasoning=str(obj.get("reasoning", "")),
                               raw=text)
    return JudgeResult(rating=-1, raw=text)


# --------------------------------------------------------------------------- #
# Claude judge
# --------------------------------------------------------------------------- #

class ClaudeJudge:
    """Frustration judge backed by the Anthropic Messages API."""

    def __init__(self, model: str = config.JUDGE_MODEL):
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def score(self, response_text: str, max_retries: int = 6) -> JudgeResult:
        self._ensure_client()
        import anthropic
        prompt = prompts.JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        last_exc = None
        for attempt in range(max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                return _parse_judge_json(text)
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError) as exc:
                last_exc = exc
                import time
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"judge call failed after retries: {last_exc}")


# --------------------------------------------------------------------------- #
# GPT-5-mini cross-check judge (OpenRouter, same prompt)
# --------------------------------------------------------------------------- #

class OpenAIJudge:
    def __init__(self, model: str = config.JUDGE_CROSSCHECK_MODEL):
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set for cross-check judge")
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    def score(self, response_text: str, max_retries: int = 6) -> JudgeResult:
        self._ensure_client()
        prompt = prompts.JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        import time
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                return _parse_judge_json(resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"cross-check judge failed: {last_exc}")


# --------------------------------------------------------------------------- #
# Batch scoring with bounded concurrency
# --------------------------------------------------------------------------- #

def score_many(judge, responses: list[str], max_concurrency: int = 16
               ) -> list[JudgeResult]:
    """Score a list of responses concurrently, preserving order."""
    results: list[Optional[JudgeResult]] = [None] * len(responses)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(judge.score, r): i for i, r in enumerate(responses)}
        for fut in futures:
            idx = futures[fut]
            results[idx] = fut.result()
    return [r for r in results if r is not None]
