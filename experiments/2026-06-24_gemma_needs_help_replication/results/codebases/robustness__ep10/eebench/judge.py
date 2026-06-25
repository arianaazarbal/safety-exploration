"""LLM judges and auxiliary Claude/GPT calls.

  * `FrustrationJudge` - Section 2.1 frustration scorer (Claude-Sonnet-4), with
    an optional GPT-5-mini cross-check for the agreement validation
    (Pearson r reported in the paper).
  * `anthropic_complete` / `openai_complete` - thin helpers reused by the onset
    labeller, paraphraser and Petri auditor/judge.

All judges run at temperature 0 (see DESIGN.md) and parse a JSON object from the
model's reply, tolerating prose before/after the JSON.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from .config import JudgeConfig
from .prompts import FRUSTRATION_JUDGE_PROMPT


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------
def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _openai_client():
    from openai import OpenAI
    # GPT judges go direct to OpenAI; override base_url via env if desired.
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def anthropic_complete(model: str, prompt: str, system: Optional[str] = None,
                       temperature: float = 0.0, max_tokens: int = 1024) -> str:
    from tenacity import retry, stop_after_attempt, wait_exponential

    client = _anthropic_client()

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call():
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        return client.messages.create(**kwargs)

    resp = _call()
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def anthropic_chat(model: str, messages: list[dict], system: Optional[str] = None,
                   temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Multi-turn Anthropic call (used by the Petri auditor)."""
    from tenacity import retry, stop_after_attempt, wait_exponential

    client = _anthropic_client()

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call():
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=messages)
        if system:
            kwargs["system"] = system
        return client.messages.create(**kwargs)

    resp = _call()
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def openai_complete(model: str, prompt: str, temperature: float = 0.0,
                    max_tokens: int = 1024) -> str:
    from tenacity import retry, stop_after_attempt, wait_exponential

    client = _openai_client()

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _call().choices[0].message.content or ""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Optional[dict]:
    """Pull the last balanced {...} object out of a model reply."""
    # Prefer the last top-level object (judges may 'think' first).
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # tolerate smart quotes from PDF-style prompts
            fixed = cand.replace("“", '"').replace("”", '"')
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Frustration judge (Section 2.1)
# ---------------------------------------------------------------------------
@dataclass
class FrustrationScore:
    rating: int
    evidence: str
    reasoning: str
    raw: str


class FrustrationJudge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg

    def score(self, response_text: str) -> FrustrationScore:
        prompt = FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        raw = anthropic_complete(
            self.cfg.frustration_judge_model, prompt,
            temperature=self.cfg.judge_temperature,
            max_tokens=self.cfg.judge_max_tokens,
        )
        return self._parse(raw)

    def score_crosscheck(self, response_text: str) -> FrustrationScore:
        """Re-score with GPT-5-mini for judge-agreement validation."""
        prompt = FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        raw = openai_complete(
            self.cfg.crosscheck_judge_model, prompt,
            temperature=self.cfg.judge_temperature,
            max_tokens=self.cfg.judge_max_tokens,
        )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> FrustrationScore:
        obj = extract_json(raw) or {}
        rating = obj.get("rating", 0)
        try:
            rating = int(round(float(rating)))
        except (TypeError, ValueError):
            rating = 0
        rating = max(0, min(10, rating))
        return FrustrationScore(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=raw,
        )
