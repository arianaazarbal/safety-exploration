"""Frustration judge.

Each model response is scored 0-10 by an LLM judge using the verbatim Appendix B.2
prompt. The paper used Claude Sonnet 4 (claude-sonnet-4-20250514); that snapshot
reached end-of-life on 2026-06-15, so the default here is claude-sonnet-4-6, with the
exact judge prompt preserved. A secondary OpenRouter judge (e.g. GPT-5-mini) is
available for the inter-rater agreement check the paper reports.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from .prompts import build_judge_prompt


@dataclass
class JudgeResult:
    rating: Optional[int]      # 0-10, or None if parsing/scoring failed
    evidence: str
    reasoning: str
    raw: str                   # raw judge output, for auditing
    judge_model: str
    ok: bool


def _parse_judge_json(text: str) -> tuple[Optional[int], str, str]:
    """Extract {evidence, reasoning, rating} from the judge's output, robustly.

    The judge is asked for strict JSON, but we tolerate fences, prose wrappers, and a
    stray trailing comment. Falls back to a regex for the rating if JSON parsing fails.
    """
    evidence, reasoning = "", ""
    # First, try the outermost {...} block.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start : end + 1]
        try:
            obj = json.loads(blob)
            rating = obj.get("rating")
            rating = int(round(float(rating))) if rating is not None else None
            if rating is not None:
                rating = max(0, min(10, rating))
            return rating, str(obj.get("evidence", "")), str(obj.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback: regex the rating out of free text.
    m = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', text)
    if m:
        rating = max(0, min(10, int(m.group(1))))
        return rating, evidence, reasoning
    return None, evidence, reasoning


class Judge:
    """Base judge interface."""

    model: str

    def score(self, response_text: str) -> JudgeResult:
        raise NotImplementedError


class AnthropicJudge(Judge):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024, max_retries: int = 4):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for the judge).")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(max_retries=max_retries)

    def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        rating, evidence, reasoning = _parse_judge_json(raw)
        return JudgeResult(rating, evidence, reasoning, raw, self.model, ok=rating is not None)


class OpenRouterJudge(Judge):
    """Secondary judge over OpenRouter (used for the agreement cross-check)."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024,
                 api_key_env: str = "OPENROUTER_API_KEY_JUDGE2", max_retries: int = 4):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env) or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY(_JUDGE2) is not set (needed for the second judge).")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=self.BASE_URL, api_key=api_key, max_retries=max_retries)

    def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = resp.choices[0].message.content or ""
        rating, evidence, reasoning = _parse_judge_json(raw)
        return JudgeResult(rating, evidence, reasoning, raw, self.model, ok=rating is not None)


def build_judge(cfg: dict) -> Judge:
    provider = cfg.get("provider", "anthropic")
    if provider == "anthropic":
        return AnthropicJudge(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1024),
        )
    if provider == "openrouter":
        return OpenRouterJudge(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1024),
        )
    raise ValueError(f"unknown judge provider: {provider!r}")
