"""OpenAI provider.

Uses the official ``openai`` SDK chat completions with JSON-schema structured
output (``response_format`` = ``json_schema``). Model id comes from config — verify
the exact id available to your account.
"""

from __future__ import annotations

from typing import Any

from .base import ModelResponse, Provider


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        import openai  # imported lazily

        self._client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.message.content or ""
        parsed = self._safe_json_loads(text) if json_schema is not None else None

        usage = {}
        if completion.usage is not None:
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }

        return ModelResponse(
            provider=self.name,
            model=self.model,
            text=text,
            parsed=parsed,
            usage=usage,
            raw=completion,
        )
