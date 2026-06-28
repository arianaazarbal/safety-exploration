"""Google (Gemini) backend — stub.

Implement against the `google-genai` SDK. The structured path should set
`response_mime_type="application/json"` and `response_schema` from the Pydantic model,
then validate the returned JSON. Left unimplemented intentionally; raises on use.
"""

from __future__ import annotations

from typing import List, Optional, Type

from pydantic import BaseModel

from .base import Message, ModelProvider, ProviderResponse


class GoogleProvider(ModelProvider):
    name = "google"

    def generate(
        self,
        system: str,
        messages: List[Message],
        *,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ProviderResponse:
        raise NotImplementedError(
            "GoogleProvider is a stub. Implement it against the google-genai SDK "
            "following the ModelProvider interface in base.py."
        )
