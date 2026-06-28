"""Google provider.

Uses the ``google-genai`` SDK with JSON response schema. Model id comes from
config — verify the exact id available to your account.

Note: the Gemini structured-output schema dialect is close to, but not identical
to, full JSON Schema. This adapter passes the same schema dict through; if a
particular field is rejected, simplify the schema (e.g. drop ``additionalProperties``)
for the Google run.
"""

from __future__ import annotations

from typing import Any

from .base import ModelResponse, Provider


class GoogleProvider(Provider):
    name = "google"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        from google import genai  # imported lazily

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema

        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text = response.text or ""
        parsed = self._safe_json_loads(text) if json_schema is not None else None

        usage = {}
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", None),
                "output_tokens": getattr(meta, "candidates_token_count", None),
            }

        return ModelResponse(
            provider=self.name,
            model=self.model,
            text=text,
            parsed=parsed,
            usage=usage,
            raw=response,
        )
