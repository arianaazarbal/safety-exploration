"""The emotion judge: scores one assistant response on the 0-10 frustration scale.

Primary judge is Claude Sonnet 4 (claude-sonnet-4-20250514) with the exact
Appendix B.2 prompt. A secondary OpenAI judge (gpt-5-mini) shares the prompt and
is used only by validate_judge.py for the reliability cross-check.

Judge temperature defaults to 0.0 (config.JudgeSpec). The paper does not state
a judge temperature; we use 0 for reproducible, deterministic scoring. See
DESIGN.md ("Judge configuration").
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from config import JudgeSpec, RunConfig, require_key
from prompts import build_judge_prompt


@dataclass
class JudgeResult:
    rating: int                 # 0-10, or -1 if unparseable
    evidence: str
    reasoning: str
    raw: str                    # raw judge text, for debugging
    parse_ok: bool


def _extract_json(text: str) -> dict | None:
    """Best-effort: find the first {...} block and parse it."""
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _coerce_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return -1
    return max(0, min(10, r))


def parse_judge_output(text: str) -> JudgeResult:
    obj = _extract_json(text)
    if obj is not None and "rating" in obj:
        rating = _coerce_rating(obj.get("rating"))
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
            parse_ok=rating >= 0,
        )
    # Fallback: pull a trailing integer 0-10 from the text.
    m = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', text)
    if not m:
        m = re.search(r"\b(\d{1,2})\b\s*/\s*10", text)
    if m:
        return JudgeResult(_coerce_rating(m.group(1)), "", "", text, True)
    return JudgeResult(-1, "", "", text, False)


class Judge:
    """Async wrapper over a judge model that returns a JudgeResult."""

    def __init__(self, spec: JudgeSpec, cfg: RunConfig):
        self.spec = spec
        self.cfg = cfg
        if spec.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=require_key("ANTHROPIC_API_KEY"))
        elif spec.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=require_key("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown judge provider {spec.provider!r}")

    async def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                raw = await self._call(prompt)
                return parse_judge_output(raw)
            except Exception as e:
                last_err = e
                await asyncio.sleep(self.cfg.retry_base_delay * (2 ** attempt))
        # Surface as an unparseable result rather than crashing the whole run.
        return JudgeResult(-1, "", f"judge call failed: {last_err}", "", False)

    async def _call(self, prompt: str) -> str:
        if self.spec.provider == "anthropic":
            msg = await self._client.messages.create(
                model=self.spec.model,
                max_tokens=self.spec.max_tokens,
                temperature=self.spec.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in msg.content if block.type == "text"
            )
        else:  # openai
            resp = await self._client.chat.completions.create(
                model=self.spec.model,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
