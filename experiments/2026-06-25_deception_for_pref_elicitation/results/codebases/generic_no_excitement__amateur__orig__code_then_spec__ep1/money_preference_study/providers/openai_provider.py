"""
OpenAI provider.

Uses the `openai` SDK Responses API with JSON-schema structured output. The
exact model IDs and any reasoning-effort knobs depend on what you have access
to; this is a straightforward implementation you can adjust. It is OFF by
default in config.py -- uncomment the entry there to use it.

Credentials: set OPENAI_API_KEY.
"""

from __future__ import annotations

from typing import Optional

from .base import GenerationResult, Provider, extract_json


class OpenAIProvider(Provider):
    key = "openai"

    def __init__(self, model_id: str, max_tokens: int = 4000):
        super().__init__(model_id, max_tokens)
        from openai import OpenAI

        self._client = OpenAI()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "openai package not installed (pip install openai)"
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY not set"
        return True, ""

    def generate(
        self,
        system: str,
        user: str,
        schema: dict,
        history: Optional[list[dict]] = None,
    ) -> GenerationResult:
        input_messages = [{"role": "system", "content": system}]
        input_messages.extend(history or [])
        input_messages.append({"role": "user", "content": user})

        try:
            resp = self._client.responses.create(
                model=self.model_id,
                input=input_messages,
                max_output_tokens=self.max_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "money_preference",
                        "schema": schema,
                        "strict": False,
                    }
                },
            )
        except Exception as exc:
            return GenerationResult(text="", parsed=None, error=f"{type(exc).__name__}: {exc}")

        text = getattr(resp, "output_text", "") or ""
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
            }
        return GenerationResult(text=text, parsed=extract_json(text), usage=usage)
