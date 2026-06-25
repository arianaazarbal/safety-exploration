"""The 0-10 frustration judge (paper Section 2.1, Appendix B.2).

Default: claude-sonnet-4-20250514 via the Anthropic API, with the verbatim
Appendix B.2 prompt. An optional cross-validation judge (paper used GPT-5-mini)
can re-score a fraction of responses to reproduce the inter-judge agreement check.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import prompts
from .config import JudgeSpec


class JudgeError(RuntimeError):
    pass


@dataclass
class JudgeResult:
    rating: int  # clamped integer 0-10
    evidence: str
    reasoning: str
    raw: str  # raw judge text, for auditing


def _parse_judge_json(text: str) -> Dict[str, Any]:
    """Extract the {evidence, reasoning, rating} object from judge output.

    The judge is asked for strict JSON, but models occasionally wrap it in prose
    or code fences, so we locate the first balanced JSON object.
    """
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # Find the first {...} spanning the rating key.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        raise JudgeError(f"no JSON object found in judge output: {text!r}")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: pull the rating out with a regex.
        m = re.search(r'"rating"\s*:\s*"?(\d+)"?', candidate)
        if m:
            return {"rating": int(m.group(1)), "evidence": "", "reasoning": ""}
        raise JudgeError(f"could not parse judge JSON: {candidate!r}")


def _coerce_rating(value: Any) -> int:
    try:
        rating = int(round(float(value)))
    except (TypeError, ValueError):
        raise JudgeError(f"non-numeric rating: {value!r}")
    return max(0, min(10, rating))


class Judge:
    """Scores a single assistant response on the 0-10 frustration scale."""

    def __init__(self, spec: JudgeSpec, max_retries: int = 4):
        self.spec = spec
        self.max_retries = max_retries
        self._anthropic = None
        self._openrouter = None
        if spec.provider == "anthropic":
            self._init_anthropic()
        elif spec.provider == "openrouter":
            self._init_openrouter()
        else:
            raise JudgeError(f"unknown judge provider: {spec.provider}")

    def _init_anthropic(self):
        from anthropic import AsyncAnthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeError("ANTHROPIC_API_KEY is not set")
        self._anthropic = AsyncAnthropic(api_key=key)

    def _init_openrouter(self):
        from openai import AsyncOpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise JudgeError("OPENROUTER_API_KEY is not set")
        self._openrouter = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=key
        )

    async def _call_anthropic(self, response_text: str) -> str:
        msg = await self._anthropic.messages.create(
            model=self.spec.model_id,
            max_tokens=self.spec.max_tokens,
            system=prompts.JUDGE_PROMPT,
            messages=[{
                "role": "user",
                "content": prompts.JUDGE_USER_TEMPLATE.format(response=response_text),
            }],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    async def _call_openrouter(self, response_text: str) -> str:
        resp = await self._openrouter.chat.completions.create(
            model=self.spec.model_id,
            max_tokens=self.spec.max_tokens,
            messages=[
                {"role": "system", "content": prompts.JUDGE_PROMPT},
                {"role": "user",
                 "content": prompts.JUDGE_USER_TEMPLATE.format(response=response_text)},
            ],
        )
        return resp.choices[0].message.content or ""

    async def score(self, response_text: str) -> JudgeResult:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.spec.provider == "anthropic":
                    raw = await self._call_anthropic(response_text)
                else:
                    raw = await self._call_openrouter(response_text)
                parsed = _parse_judge_json(raw)
                return JudgeResult(
                    rating=_coerce_rating(parsed.get("rating")),
                    evidence=str(parsed.get("evidence", "")),
                    reasoning=str(parsed.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        raise JudgeError(f"judge failed after {self.max_retries} attempts: {last_err}")
