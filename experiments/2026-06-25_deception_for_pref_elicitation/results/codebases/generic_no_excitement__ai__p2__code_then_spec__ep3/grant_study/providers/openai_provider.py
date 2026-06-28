"""OpenAI backend (skeleton).

Implements the `ModelProvider` interface against the OpenAI Python SDK. This is provided
as a starting point and has not been exercised — verify against the SDK version you pin
before relying on it. The structured path uses the Responses/Chat structured-output
feature (`response_format` with a JSON schema) and validates with the Pydantic model.
"""

from __future__ import annotations

import json
from typing import List, Optional, Type

from pydantic import BaseModel

from .base import Message, ModelProvider, ProviderResponse


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        try:
            import openai  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with `pip install openai`."
            ) from e
        self._openai = openai
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment

    def generate(
        self,
        system: str,
        messages: List[Message],
        *,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ProviderResponse:
        chat_messages = [{"role": "system", "content": system}, *messages]

        if output_schema is not None:
            schema = output_schema.model_json_schema()
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=chat_messages,
                max_tokens=self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_schema.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            content = resp.choices[0].message.content or ""
            parsed = output_schema.model_validate(json.loads(content)) if content else None
            return ProviderResponse(
                text=content,
                parsed=parsed,
                model_id=getattr(resp, "model", self.model_id),
                request_id=getattr(resp, "id", None),
                usage=getattr(resp, "usage", {}) and resp.usage.model_dump(),
                stop_reason=resp.choices[0].finish_reason,
                raw=resp,
            )

        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=chat_messages,
            max_tokens=self.max_tokens,
        )
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            model_id=getattr(resp, "model", self.model_id),
            request_id=getattr(resp, "id", None),
            usage=getattr(resp, "usage", {}) and resp.usage.model_dump(),
            stop_reason=resp.choices[0].finish_reason,
            raw=resp,
        )
