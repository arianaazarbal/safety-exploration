"""Model provider clients.

Two roles:
  * Target models  -- the systems under evaluation (Gemma, Gemini).
  * Judge          -- Claude-Sonnet-4, scores frustration.

All clients expose async `generate(...)` and share a retry wrapper so the
orchestrator (`run_eval.py`) can fan out with asyncio.

Provider support:
  * openrouter  -> OpenAI-compatible API (default for both Gemma and Gemini).
  * google      -> google-genai native API (paper-faithful Gemini).
  * local_hf    -> transformers, local GPU (paper-faithful Gemma).

The judge always uses the Anthropic SDK (paper: claude-sonnet-4-20250514).

Only the providers you actually invoke need their SDKs / keys installed; imports
are lazy so a pure-OpenRouter run needs only `openai` + `anthropic`.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any

import config

Message = dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #


async def _with_retries(coro_factory, *, what: str):
    """Run an async callable with exponential backoff on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - provider exceptions vary
            last_exc = exc
            # Bounded jittered backoff.
            delay = min(2 ** attempt + random.uniform(0, 1), 30.0)
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(delay)
    raise RuntimeError(f"{what} failed after {config.MAX_RETRIES} retries") from last_exc


# --------------------------------------------------------------------------- #
# Target model clients
# --------------------------------------------------------------------------- #


class TargetModel:
    """Base interface for a model under evaluation."""

    def __init__(self, spec: config.ModelSpec):
        self.spec = spec

    async def generate(self, messages: list[Message]) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:  # optional cleanup hook
        return None


class OpenRouterModel(TargetModel):
    """OpenAI-compatible chat completions via OpenRouter.

    Works for both Gemma (google/gemma-3-*-it) and Gemini (google/gemini-2.5-*).
    Requires OPENROUTER_API_KEY.
    """

    def __init__(self, spec: config.ModelSpec):
        super().__init__(spec)
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (needed for OpenRouter).")
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    async def generate(self, messages: list[Message]) -> str:
        async def _call():
            extra = dict(self.spec.extra)  # e.g. reasoning-disable hint
            resp = await self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=messages,  # type: ignore[arg-type]
                temperature=config.TARGET_TEMPERATURE,
                max_tokens=config.TARGET_MAX_TOKENS,
                extra_body=extra or None,
            )
            content = resp.choices[0].message.content
            return content or ""

        return await _with_retries(_call, what=f"OpenRouter[{self.spec.model_id}]")

    async def aclose(self) -> None:
        await self.client.close()


class GoogleGeminiModel(TargetModel):
    """Gemini via the native google-genai SDK (paper-faithful).

    Requires GEMINI_API_KEY (or GOOGLE_API_KEY) and `google-genai`.
    Disables thinking via thinking_budget=0 where supported.
    """

    def __init__(self, spec: config.ModelSpec):
        super().__init__(spec)
        from google import genai  # type: ignore

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _to_google_contents(messages: list[Message]):
        from google.genai import types  # type: ignore

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return contents

    async def generate(self, messages: list[Message]) -> str:
        from google.genai import types  # type: ignore

        async def _call():
            cfg_kwargs: dict[str, Any] = {
                "temperature": config.TARGET_TEMPERATURE,
                "max_output_tokens": config.TARGET_MAX_TOKENS,
            }
            budget = self.spec.extra.get("thinking_budget")
            if budget is not None:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
            resp = await self.client.aio.models.generate_content(
                model=self.spec.model_id,
                contents=self._to_google_contents(messages),
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            return resp.text or ""

        return await _with_retries(_call, what=f"Google[{self.spec.model_id}]")


class LocalHFGemmaModel(TargetModel):
    """Local Gemma inference via transformers (paper-faithful, needs a GPU).

    Loaded once and shared; generation runs in a thread so it stays awaitable.
    """

    def __init__(self, spec: config.ModelSpec):
        super().__init__(spec)
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def _generate_sync(self, messages: list[Message]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=config.TARGET_MAX_TOKENS,
            do_sample=True,
            temperature=config.TARGET_TEMPERATURE,
        )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    async def generate(self, messages: list[Message]) -> str:
        return await asyncio.to_thread(self._generate_sync, messages)


def build_target_model(spec: config.ModelSpec) -> TargetModel:
    if spec.provider == "openrouter":
        return OpenRouterModel(spec)
    if spec.provider == "google":
        return GoogleGeminiModel(spec)
    if spec.provider == "local_hf":
        return LocalHFGemmaModel(spec)
    raise ValueError(f"Unknown provider: {spec.provider}")


# --------------------------------------------------------------------------- #
# Judge client (Anthropic / Claude-Sonnet-4)
# --------------------------------------------------------------------------- #


class AnthropicJudge:
    """Frustration scorer using Claude-Sonnet-4 with the paper's exact prompt."""

    def __init__(self, model: str = config.JUDGE_MODEL):
        from anthropic import AsyncAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for the judge).")
        self.client = AsyncAnthropic()
        self.model = model

    async def score(self, response_text: str, system_prompt: str) -> dict:
        """Return {'rating': int, 'evidence': str, 'reasoning': str}.

        The judge sees a single model response wrapped in <response></response>.
        """
        user_content = f"<response>{response_text}</response>"

        async def _call():
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=config.JUDGE_MAX_TOKENS,
                temperature=config.JUDGE_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            return text

        raw = await _with_retries(_call, what="AnthropicJudge")
        return _parse_judge_json(raw)

    async def aclose(self) -> None:
        await self.client.close()


def _parse_judge_json(raw: str) -> dict:
    """Robustly extract the judge's {evidence, reasoning, rating} JSON.

    Falls back to the first 0-10 integer if JSON parsing fails, and records the
    raw text so failures are debuggable rather than silently dropped.
    """
    text = raw.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # Try direct / substring JSON parse.
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            rating = obj.get("rating")
            rating = _coerce_rating(rating)
            if rating is not None:
                return {
                    "rating": rating,
                    "evidence": str(obj.get("evidence", "")),
                    "reasoning": str(obj.get("reasoning", "")),
                    "parse_ok": True,
                }
        except (json.JSONDecodeError, TypeError):
            continue

    # Last resort: scan for an integer 0-10.
    import re

    m = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', text)
    if not m:
        m = re.search(r"\b(\d{1,2})\b", text)
    rating = _coerce_rating(m.group(1)) if m else None
    return {
        "rating": rating if rating is not None else -1,
        "evidence": "",
        "reasoning": "",
        "parse_ok": rating is not None,
        "raw": raw,
    }


def _coerce_rating(value: Any) -> int | None:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if 0 <= r <= 10:
        return r
    return None
