"""LLM-judge / auditor clients (Claude, and an optional GPT validator).

`ClaudeClient` is a thin Anthropic wrapper used for:
  - the frustration judge (Section 2.1, claude-sonnet-4)
  - emotion-onset labelling + paraphrasing (Section 3.1, claude-sonnet-4)
  - the Petri auditor (claude-sonnet-4) and judge (claude-opus-4)

`OpenAIJudgeClient` wraps GPT-5-mini for the judge-agreement validation
(Section 2.1: Pearson r between Claude and GPT scores).

Both expose `complete(system, user)` returning text, with retry/backoff. JSON
parsing lives in the callers (scoring.py / petri).
"""
from __future__ import annotations

import time

import config


class ClaudeClient:
    def __init__(self, model: str = config.JUDGE_MODEL,
                 temperature: float = config.JUDGE_TEMPERATURE,
                 max_tokens: int = config.JUDGE_MAX_TOKENS):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, user: str, system: str | None = None,
                 max_tokens: int | None = None, attempt: int = 0) -> str:
        try:
            kwargs = dict(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": user}],
            )
            if system:
                kwargs["system"] = system
            resp = self.client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception:
            if attempt >= 5:
                raise
            time.sleep(2 ** attempt)
            return self.complete(user, system, max_tokens, attempt + 1)

    def chat(self, messages: list[dict], system: str | None = None,
             max_tokens: int | None = None, attempt: int = 0) -> str:
        """Multi-turn variant for the Petri auditor loop."""
        try:
            kwargs = dict(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=messages,
            )
            if system:
                kwargs["system"] = system
            resp = self.client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception:
            if attempt >= 5:
                raise
            time.sleep(2 ** attempt)
            return self.chat(messages, system, max_tokens, attempt + 1)


class OpenAIJudgeClient:
    def __init__(self, model: str = config.VALIDATION_JUDGE_MODEL):
        from openai import OpenAI

        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model

    def complete(self, user: str, system: str | None = None,
                 attempt: int = 0) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt >= 5:
                raise
            time.sleep(2 ** attempt)
            return self.complete(user, system, attempt + 1)
