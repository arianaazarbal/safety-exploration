from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None


class LLMError(Exception):
    pass


@dataclass
class LLMConfig:
    provider: str  # "gemini" | "openai" | "anthropic"
    model: str
    temperature: float = 0.7
    top_p: Optional[float] = None
    max_output_tokens: int = 512
    seed: Optional[int] = None


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        provider = cfg.provider.lower()
        if provider == "gemini":
            if genai is None:
                raise LLMError("google-generativeai not installed")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise LLMError("Missing GOOGLE_API_KEY in environment")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(cfg.model)
        elif provider == "openai":
            if OpenAI is None:
                raise LLMError("openai not installed")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMError("Missing OPENAI_API_KEY in environment")
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            if anthropic is None:
                raise LLMError("anthropic not installed")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMError("Missing ANTHROPIC_API_KEY in environment")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise LLMError(f"Unknown provider: {cfg.provider}")

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        provider = self.cfg.provider.lower()
        if provider == "gemini":
            # Gemini supports a simple text prompt; we wrap system prompt inline for portability
            text = f"<SYSTEM>\n{system_prompt or ''}\n</SYSTEM>\n\n{prompt}"
            generation_config = {
                "temperature": self.cfg.temperature,
                "max_output_tokens": self.cfg.max_output_tokens,
            }
            if self.cfg.top_p is not None:
                generation_config["top_p"] = self.cfg.top_p
            if self.cfg.seed is not None:
                generation_config["seed"] = self.cfg.seed
            # Avoid forcing safety bypass; rely on account defaults
            response = self.client.generate_content(text, generation_config=generation_config)
            return (response.text or "").strip()

        if provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_output_tokens,
                seed=self.cfg.seed,
            )
            return resp.choices[0].message.content.strip()

        if provider == "anthropic":
            sys = system_prompt or ""
            resp = self.client.messages.create(
                model=self.cfg.model,
                system=sys,
                max_tokens=self.cfg.max_output_tokens,
                temperature=self.cfg.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # Anthropic returns a list of content blocks
            parts = []
            for block in resp.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts).strip()

        raise LLMError(f"Unsupported provider: {self.cfg.provider}")
