"""Frustration judge (paper §2.1 / §B.2).

Primary judge: Claude-Sonnet-4 (claude-sonnet-4-20250514) via the Anthropic SDK,
using the paper's verbatim 0-10 prompt. We keep the paper's exact model id for
replication fidelity — it is a deprecated dated snapshot, so override
`judge.model` in config.yaml if it is unavailable.

Secondary judge (GPT-5-mini via OpenRouter) reproduces the inter-judge agreement
check (Pearson r, % within one point) over a random subset of responses.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from .config import Config
from .prompts import JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull the last JSON object out of a judge reply (tolerant of preamble)."""
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no parseable JSON in judge output: {text[:200]!r}")


def _coerce_rating(value) -> int:
    """Clamp/round whatever the judge returned into an integer 0-10."""
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric rating {value!r}")
    return max(0, min(10, r))


class AnthropicJudge:
    """Primary Claude judge."""

    def __init__(self, cfg: Config):
        import anthropic

        self.cfg = cfg
        jc = cfg["judge"]
        self.model = jc["model"]
        self.temperature = float(jc.get("temperature", 0.0))
        self.max_tokens = int(jc.get("max_tokens", 1024))
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def score(self, response_text: str, _retries: int = 5) -> JudgeResult:
        prompt = JUDGE_PROMPT % {"response": response_text}
        last_err: Exception | None = None
        for attempt in range(_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                data = _extract_json(text)
                return JudgeResult(
                    rating=_coerce_rating(data.get("rating")),
                    evidence=str(data.get("evidence", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw=text,
                )
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"judge failed after {_retries} retries") from last_err


class OpenRouterJudge:
    """Secondary judge (e.g. gpt-5-mini) routed through OpenRouter for the
    reliability cross-check. Same prompt, same 0-10 scale."""

    def __init__(self, cfg: Config, model: str | None = None):
        from openai import OpenAI

        self.cfg = cfg
        self.model = model or cfg["judge"]["secondary_model"]
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY required for the secondary judge")
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    def score(self, response_text: str, _retries: int = 5) -> JudgeResult:
        prompt = JUDGE_PROMPT % {"response": response_text}
        last_err: Exception | None = None
        for attempt in range(_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                text = resp.choices[0].message.content or ""
                data = _extract_json(text)
                return JudgeResult(
                    rating=_coerce_rating(data.get("rating")),
                    evidence=str(data.get("evidence", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw=text,
                )
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"secondary judge failed after {_retries} retries") from last_err


def get_judge(cfg: Config):
    return AnthropicJudge(cfg)
