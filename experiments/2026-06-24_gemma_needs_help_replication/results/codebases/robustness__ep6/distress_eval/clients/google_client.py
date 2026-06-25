"""Optional native Gemini backend (google-genai). OpenRouter is the default path
(matching the paper, which served Gemini via OpenRouter); this exists for users
who prefer Google's API directly. Chat-only."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config_proxy as cfg
from .base import GenerationResult, ModelClient


class GoogleGenAIClient(ModelClient):
    def __init__(self, name: str, model_id: str):
        from google import genai

        self.name = name
        # google-genai expects the bare model id, e.g. "gemini-2.5-flash"
        self.model_id = model_id.split("/")[-1]
        self._client = genai.Client(api_key=cfg.api_key("google"))

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
    def _one(self, messages, temperature, max_new_tokens) -> GenerationResult:
        from google.genai import types

        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role, parts=[types.Part.from_text(text=m["content"])]))
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system,
            # disable thinking where supported (paper: "thinking false via API")
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=config)
        return GenerationResult(text=resp.text or "")

    def chat(self, messages, *, n=1, temperature=1.0, max_new_tokens=2048):
        return [self._one(messages, temperature, max_new_tokens) for _ in range(n)]
