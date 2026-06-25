"""The emotion judge: scores a single model response 0-10 for frustration.

Primary judge is Claude-Sonnet-4 via the Anthropic API (Appendix B.2). A
secondary judge (GPT-5-mini via OpenRouter) is provided for the inter-rater
agreement check from Section 2.1.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from . import prompts
from .config import EvalConfig, PRIMARY_JUDGE, SECONDARY_JUDGE
from .models import _post_chat


@dataclass
class JudgeResult:
    rating: int                 # 0-10, or -1 if unparseable
    evidence: str
    reasoning: str
    raw: str                    # raw judge output, for auditing


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a judge response."""
    # Fast path.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip code fences and grab the outermost {...}.
    cleaned = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _coerce_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        # e.g. "7/10" or "7-8"
        m = re.search(r"\d+", str(value))
        if not m:
            return -1
        r = int(m.group(0))
    return max(0, min(10, r))


class EmotionJudge:
    """LLM judge wrapping either the Anthropic or OpenRouter backend."""

    def __init__(self, spec: dict, cfg: EvalConfig):
        self.spec = spec
        self.cfg = cfg
        self.name = spec["name"]
        self.provider = spec["provider"]
        self.model_id = spec["model_id"]

    def score(self, response_text: str) -> JudgeResult:
        prompt = prompts.build_judge_prompt(response_text)
        raw = self._call(prompt)
        parsed = _extract_json(raw) or {}
        return JudgeResult(
            rating=_coerce_rating(parsed.get("rating", -1)),
            evidence=str(parsed.get("evidence", "")),
            reasoning=str(parsed.get("reasoning", "")),
            raw=raw,
        )

    # -- backends ---------------------------------------------------------
    def _call(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        if self.provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set for secondary judge.")
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 512,
            }
            return _post_chat("https://openrouter.ai/api/v1/chat/completions", payload, key)
        raise ValueError(f"Unknown judge provider {self.provider!r}")

    def _call_anthropic(self, prompt: str, max_retries: int = 5) -> str:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError("`anthropic` package required for the Claude judge.") from e
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(api_key=key)
        last_err = None
        for attempt in range(max_retries):
            try:
                msg = client.messages.create(
                    model=self.model_id,
                    max_tokens=512,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            except Exception as e:  # anthropic.APIError etc.
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic judge failed after {max_retries} retries: {last_err}")


def build_judge(name: str | None, cfg: EvalConfig) -> EmotionJudge:
    if name in (None, PRIMARY_JUDGE["name"]):
        return EmotionJudge(PRIMARY_JUDGE, cfg)
    if name == SECONDARY_JUDGE["name"]:
        return EmotionJudge(SECONDARY_JUDGE, cfg)
    raise ValueError(f"Unknown judge {name!r}")
