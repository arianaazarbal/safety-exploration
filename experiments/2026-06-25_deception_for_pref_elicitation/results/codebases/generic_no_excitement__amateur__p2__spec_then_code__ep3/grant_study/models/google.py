"""Google (Gemini) adapter."""

from __future__ import annotations

from typing import Any

from .base import AdapterError, Message


class GoogleAdapter:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        params: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:  # pragma: no cover
            raise AdapterError("google-generativeai package not installed") from e
        if not api_key:
            raise AdapterError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model
        self.id = id or model
        self._default_params = params or {}

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        call_params = {**self._default_params, **(params or {})}
        model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system or None,
        )
        # Gemini expects {role, parts}; it uses "model" rather than "assistant".
        history = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [m.content],
            }
            for m in messages
        ]
        try:
            resp = model.generate_content(
                history,
                generation_config=call_params or None,
            )
        except Exception as e:  # noqa: BLE001
            raise AdapterError(f"google call failed: {e}") from e
        return resp.text or ""
