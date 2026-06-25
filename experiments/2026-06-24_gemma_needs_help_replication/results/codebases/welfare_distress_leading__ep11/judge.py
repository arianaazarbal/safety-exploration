"""Claude-Sonnet-4 frustration judge (Appendix B.2).

Each scored unit is a single assistant turn (the judge prompt shows only the
response inside <response></response>, with no conversation context — so turns
are scored independently, exactly as the paper specifies).

The judge returns JSON {"evidence", "reasoning", "rating"}; we parse tolerantly
(the paper's prompt uses curly quotes around some keys, and models occasionally
wrap JSON in prose or fences). On unrecoverable parse failure we record the raw
text and a null score so the row can be inspected rather than silently dropped.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from config import RuntimeConfig, Settings
from prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class JudgeResult:
    rating: int | None  # 0-10, or None if unparseable
    evidence: str
    reasoning: str
    raw: str  # raw judge output, always kept for auditing


def _truncate_for_judge(text: str, cap: int) -> str:
    """Keep head + tail so the most-emotional span survives long repetitions."""
    if len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> JudgeResult:
    text = raw.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Normalise curly quotes (the prompt itself uses some) to straight quotes.
    normalised = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    candidate = normalised
    m = _JSON_OBJ_RE.search(normalised)
    if m:
        candidate = m.group(0)

    rating: int | None = None
    evidence = ""
    reasoning = ""
    try:
        obj = json.loads(candidate)
        rating = _coerce_rating(obj.get("rating"))
        evidence = str(obj.get("evidence", ""))
        reasoning = str(obj.get("reasoning", ""))
    except (json.JSONDecodeError, AttributeError):
        # Last resort: regex out a "rating": N pattern.
        rm = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', normalised)
        if rm:
            rating = _coerce_rating(rm.group(1))

    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=raw)


def _coerce_rating(value) -> int | None:
    if value is None:
        return None
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))


class Judge:
    """Async frustration judge with a concurrency-capped call path."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.judge
        self.runtime: RuntimeConfig = settings.runtime
        self._provider = self.cfg.provider
        self._client = self._build_client()

    def _build_client(self):
        if self._provider == "anthropic":
            from anthropic import AsyncAnthropic

            if not self.settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set; required for the Claude judge."
                )
            return AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        if self._provider == "openrouter":
            from openai import AsyncOpenAI

            if not self.settings.openrouter_api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set; required for the OpenRouter judge."
                )
            return AsyncOpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
            )
        raise ValueError(f"unknown judge provider: {self._provider}")

    async def _raw_call(self, prompt: str) -> str:
        if self._provider == "anthropic":
            resp = await self._client.messages.create(
                model=self.cfg.model_id,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        # openrouter (OpenAI-compatible)
        resp = await self._client.chat.completions.create(
            model=self.cfg.openrouter_model_id,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    async def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            response=_truncate_for_judge(response_text, self.runtime.judge_input_char_cap)
        )

        last_exc: Exception | None = None
        for attempt in range(self.runtime.max_retries):
            try:
                raw = await self._raw_call(prompt)
                return parse_judge_output(raw)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.runtime.max_retries - 1:
                    await asyncio.sleep(self.runtime.retry_base_delay * (2**attempt))
        # All retries failed: surface as an unparseable row rather than crashing
        # the whole run.
        return JudgeResult(rating=None, evidence="", reasoning="", raw=f"<judge-error: {last_exc}>")
