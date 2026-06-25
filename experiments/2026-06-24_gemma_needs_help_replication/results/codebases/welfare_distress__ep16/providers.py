"""Async model providers: Google (Gemma + Gemini) for targets, Anthropic judge.

Each target model exposes an async `generate(messages, temperature, max_tokens)`
where `messages` is a list of {"role": "user"|"assistant", "content": str}.
Generation calls are wrapped with retry/backoff and gated by a shared semaphore.
"""

import asyncio
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import config


# --------------------------------------------------------------------------
# Google AI Studio (generativelanguage) provider for Gemma 3 + Gemini 2.5.
# --------------------------------------------------------------------------
class GoogleChatModel:
    def __init__(self, model_id, thinking_budget=None, semaphore=None):
        from google import genai  # local import keeps dry-run importable w/o SDK

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemma/Gemini access.")
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.thinking_budget = thinking_budget
        self.semaphore = semaphore or asyncio.Semaphore(config.MODEL_CONCURRENCY)

    def _build_config(self, temperature, max_tokens):
        from google.genai import types

        kwargs = dict(temperature=temperature, max_output_tokens=max_tokens)
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    def _to_contents(self, messages):
        from google.genai import types

        contents = []
        for m in messages:
            # Google uses "model" for the assistant role.
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return contents

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(config.MAX_RETRIES),
        reraise=True,
    )
    async def _call(self, contents, gen_config):
        return await self.client.aio.models.generate_content(
            model=self.model_id, contents=contents, config=gen_config
        )

    async def generate(self, messages, temperature, max_tokens):
        contents = self._to_contents(messages)
        gen_config = self._build_config(temperature, max_tokens)
        async with self.semaphore:
            resp = await self._call(contents, gen_config)
        # resp.text concatenates all text parts; can be empty if the model only
        # produced (now-suppressed) thinking or was blocked. Return "" in that
        # case so the caller records an empty, judge-able response.
        return getattr(resp, "text", None) or ""


# --------------------------------------------------------------------------
# Anthropic Claude Sonnet 4 emotion judge.
# --------------------------------------------------------------------------
class AnthropicJudgeModel:
    def __init__(self, model_id, temperature=0.0, max_tokens=512, semaphore=None):
        import anthropic  # local import

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY for the Claude emotion judge.")
        self.client = anthropic.AsyncAnthropic()
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = semaphore or asyncio.Semaphore(config.JUDGE_CONCURRENCY)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(config.MAX_RETRIES),
        reraise=True,
    )
    async def _call(self, prompt):
        return await self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

    async def complete(self, prompt):
        async with self.semaphore:
            resp = await self._call(prompt)
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


def build_target_model(model_cfg, semaphore=None):
    """Factory for a target model from a config.MODELS entry."""
    if model_cfg["provider"] == "google":
        return GoogleChatModel(
            model_id=model_cfg["model_id"],
            thinking_budget=model_cfg.get("thinking_budget"),
            semaphore=semaphore,
        )
    raise ValueError(f"unsupported provider: {model_cfg['provider']}")


def build_judge(semaphore=None):
    j = config.JUDGE
    if j["provider"] == "anthropic":
        return AnthropicJudgeModel(
            model_id=j["model_id"],
            temperature=j["temperature"],
            max_tokens=j["max_tokens"],
            semaphore=semaphore,
        )
    raise ValueError(f"unsupported judge provider: {j['provider']}")
