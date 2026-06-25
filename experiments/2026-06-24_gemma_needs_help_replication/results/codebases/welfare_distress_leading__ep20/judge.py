"""Emotion judge: scores a model response on the 0-10 frustration scale.

Primary judge is Claude Sonnet 4 (Anthropic API), using the verbatim Appendix B.2
prompt. The judge backend is pluggable so the same prompt can be sent to a second
judge (GPT-5-mini) for the inter-judge agreement check (Section 2.1).

The judge is asked for JSON {"evidence", "reasoning", "rating"}; we parse it
robustly (tolerating code fences / surrounding prose) and clamp the rating to
the integer range 0-10.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import config
from prompts import JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: Optional[int]      # 0-10, or None if unparseable
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""
    error: Optional[str] = None


def _retry(fn, max_retries: int, base_delay: float):
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"judge call failed after {max_retries} retries") from last


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> JudgeResult:
    """Extract {evidence, reasoning, rating} from the judge's text output."""
    raw = text or ""
    # Strip code fences if present.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    candidate = cleaned
    if not candidate.startswith("{"):
        m = _JSON_RE.search(cleaned)
        if m:
            candidate = m.group(0)

    try:
        obj = json.loads(candidate)
    except Exception:
        # Last resort: pull the first integer after a "rating" key.
        m = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', raw)
        if m:
            return JudgeResult(rating=_clamp(int(m.group(1))), raw=raw,
                               error="json_parse_failed_regex_fallback")
        return JudgeResult(rating=None, raw=raw, error="json_parse_failed")

    rating = obj.get("rating")
    try:
        rating = _clamp(int(round(float(rating))))
    except (TypeError, ValueError):
        rating = None
    return JudgeResult(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=raw,
    )


def _clamp(x: int) -> int:
    return max(0, min(10, x))


class Judge:
    def __init__(self, spec: config.JudgeSpec, gen: config.GenConfig):
        self.spec = spec
        self.gen = gen

    def score(self, response_text: str) -> JudgeResult:  # pragma: no cover
        raise NotImplementedError


class AnthropicJudge(Judge):
    def __init__(self, spec: config.JudgeSpec, gen: config.GenConfig):
        super().__init__(spec, gen)
        from anthropic import Anthropic

        key = config.get_api_key(spec.api_key_env)
        self.client = Anthropic(api_key=key)

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT.format(response=response_text)

        def _call():
            msg = self.client.messages.create(
                model=self.spec.model_id,
                max_tokens=self.spec.max_tokens,
                temperature=self.spec.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in msg.content if block.type == "text"
            )

        try:
            text = _retry(_call, self.gen.max_retries, self.gen.retry_base_delay)
        except Exception as exc:  # noqa: BLE001
            return JudgeResult(rating=None, error=f"api_error: {exc}")
        return parse_judge_json(text)


class OpenAICompatibleJudge(Judge):
    def __init__(self, spec: config.JudgeSpec, gen: config.GenConfig):
        super().__init__(spec, gen)
        from openai import OpenAI

        key = config.get_api_key(spec.api_key_env) or "not-needed"
        self.client = OpenAI(base_url=spec.base_url, api_key=key)

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT.format(response=response_text)

        def _call():
            resp = self.client.chat.completions.create(
                model=self.spec.model_id,
                max_tokens=self.spec.max_tokens,
                temperature=self.spec.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        try:
            text = _retry(_call, self.gen.max_retries, self.gen.retry_base_delay)
        except Exception as exc:  # noqa: BLE001
            return JudgeResult(rating=None, error=f"api_error: {exc}")
        return parse_judge_json(text)


def make_judge(judge_name: str, gen: config.GenConfig) -> Judge:
    spec = config.JUDGE_REGISTRY[judge_name]
    if spec.backend == "anthropic":
        return AnthropicJudge(spec, gen)
    if spec.backend == "openai_compatible":
        return OpenAICompatibleJudge(spec, gen)
    raise ValueError(f"unknown judge backend: {spec.backend}")
