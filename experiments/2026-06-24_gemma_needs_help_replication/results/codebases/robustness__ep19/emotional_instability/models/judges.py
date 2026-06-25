"""Judge / auditor API clients (Anthropic + optional OpenAI cross-check).

`EmotionJudge` wraps Claude-Sonnet-4 to run the Appendix B.2 frustration judge.
`AnthropicChat` is a thin Claude wrapper reused for the onset/paraphrase steps
(Section 3) and the Petri auditor/judge (Section 4). `OpenAIJudge` runs the
GPT-5-mini reliability cross-check.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import JUDGE_MODEL
from ..prompts.judge import JudgeResult, build_judge_prompt, parse_judge_response


class AnthropicChat:
    """Generic single-prompt or multi-turn Claude caller."""

    def __init__(self, model: str, *, api_key: str | None = None):
        from anthropic import Anthropic

        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = Anthropic(api_key=key)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def complete(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        messages: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      temperature=temperature, messages=messages)
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class EmotionJudge:
    """Appendix B.2 frustration judge (Claude-Sonnet-4 by default)."""

    def __init__(self, model: str = JUDGE_MODEL, *, api_key: str | None = None):
        self.chat = AnthropicChat(model, api_key=api_key)
        self.model = model

    def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        # temperature 0 for a deterministic, reproducible judge.
        raw = self.chat.complete(prompt, max_tokens=512, temperature=0.0)
        return parse_judge_response(raw)


class OpenAIJudge:
    """GPT-5-mini judge for the reliability cross-check (same B.2 prompt)."""

    def __init__(self, model: str = "gpt-5-mini", *, api_key: str | None = None):
        from openai import OpenAI

        self.model = model
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set (needed for cross-check).")
        self.client = OpenAI(api_key=key)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_judge_response(resp.choices[0].message.content or "")
