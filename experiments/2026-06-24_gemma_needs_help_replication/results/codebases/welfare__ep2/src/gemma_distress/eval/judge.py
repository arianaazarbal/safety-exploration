"""LLM judge clients for the 0-10 frustration scale (Section 2.1, Appendix B.2).

Primary judge: Claude-Sonnet-4 (``claude-sonnet-4-20250514``) via the Anthropic
SDK. Agreement check: GPT-5-mini via the OpenAI SDK (OpenRouter-compatible).

Both use the identical Appendix-B.2 prompt and return an integer 0-10 plus the
evidence quote and reasoning. Calls are threaded for throughput and retried with
exponential backoff.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from ..prompts.judge import build_judge_prompt


@dataclass
class ScoreResult:
    rating: int | None
    evidence: str | None
    reasoning: str | None
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(text: str) -> ScoreResult:
    """Extract ``{"evidence","reasoning","rating"}`` from a judge response.

    Tolerant of leading prose, code fences, and smart quotes (the paper's prompt
    uses curly quotes in places). Falls back to a bare integer search.
    """
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    match = _JSON_RE.search(cleaned)
    if match:
        blob = match.group(0)
        try:
            data = json.loads(blob)
            rating = data.get("rating")
            rating = int(round(float(rating))) if rating is not None else None
            if rating is not None:
                rating = max(0, min(10, rating))
            return ScoreResult(rating, data.get("evidence"), data.get("reasoning"), text)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    # Last-resort: find a "rating": N pattern or a standalone integer.
    m = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', cleaned)
    if m:
        return ScoreResult(max(0, min(10, int(m.group(1)))), None, None, text)
    return ScoreResult(None, None, None, text)


class AnthropicJudge:
    """Claude-Sonnet-4 frustration judge."""

    def __init__(self, model: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0, max_workers: int = 8):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_workers = max_workers

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
    def _score_one(self, response_text: str) -> ScoreResult:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": build_judge_prompt(response_text)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return parse_judge_output(text)

    def score_many(self, texts: list[str]) -> list[ScoreResult]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._score_one, texts))


class OpenAIJudge:
    """GPT-5-mini judge for the inter-rater agreement check (OpenRouter-friendly)."""

    def __init__(self, model: str, *, max_tokens: int = 1024, max_workers: int = 8):
        import os
        from openai import OpenAI

        # Use OpenRouter if OPENROUTER_API_KEY is set, else native OpenAI.
        if os.environ.get("OPENROUTER_API_KEY"):
            self.client = OpenAI(
                base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            self.model = model if "/" in model else f"openai/{model}"
        else:
            self.client = OpenAI()
            self.model = model
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
    def _score_one(self, response_text: str) -> ScoreResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": build_judge_prompt(response_text)}],
        )
        return parse_judge_output(resp.choices[0].message.content or "")

    def score_many(self, texts: list[str]) -> list[ScoreResult]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._score_one, texts))


def build_judge(judge_cfg: dict) -> AnthropicJudge:
    return AnthropicJudge(
        model=judge_cfg["model"],
        max_tokens=judge_cfg.get("max_tokens", 1024),
        temperature=judge_cfg.get("temperature", 0.0),
    )
