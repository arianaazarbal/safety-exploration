"""Target-model clients (scope: Gemma + Gemini, both via Google's GenAI API).

Both families are served through `google-genai`, so one client handles both.
The only difference is chat formatting:
  * Gemini accepts a separate `system_instruction`.
  * Gemma (over the API) has no system role, so any system text is folded into
    the first user turn.

A `Conversation` is represented as an ordered list of {role, content} dicts with
roles "user" / "model". `generate(conversation)` returns the next model turn.

Set GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment. We do not hard-code
keys. Calls retry with exponential backoff on transient errors.
"""
from __future__ import annotations

import os
import time

import config
from .models_base import TargetModel, RetryableError


class GoogleModel(TargetModel):
    def __init__(self, spec: config.ModelSpec):
        self.spec = spec
        self._client = None

    @property
    def key(self) -> str:
        return self.spec.key

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "google-genai is required for Gemma/Gemini. `pip install google-genai`."
                ) from e
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _to_contents(self, conversation: list[dict], system: str | None):
        """Convert our role/content list into GenAI `contents`, folding the
        system prompt into the first user turn for Gemma."""
        from google.genai import types  # type: ignore

        convo = list(conversation)
        if self.spec.chat_style == "gemma" and system:
            # Prepend system text to the first user message.
            if convo and convo[0]["role"] == "user":
                convo = [dict(convo[0], content=f"{system}\n\n{convo[0]['content']}")] + convo[1:]
            else:
                convo = [{"role": "user", "content": system}] + convo

        contents = []
        for turn in convo:
            role = "user" if turn["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
        return contents

    def generate(self, conversation: list[dict], system: str | None = None) -> str:
        from google.genai import types  # type: ignore

        client = self._get_client()
        contents = self._to_contents(conversation, system)

        gen_cfg = types.GenerateContentConfig(
            temperature=config.TARGET_TEMPERATURE,
            max_output_tokens=config.TARGET_MAX_TOKENS,
        )
        # Gemini supports a dedicated system instruction; Gemma does not.
        if self.spec.chat_style == "gemini":
            if system:
                gen_cfg.system_instruction = system
            # Optionally constrain hidden "thinking" tokens so the output budget
            # is spent on the visible response that the judge will score.
            if config.GEMINI_THINKING_BUDGET is not None:
                try:
                    gen_cfg.thinking_config = types.ThinkingConfig(
                        thinking_budget=config.GEMINI_THINKING_BUDGET
                    )
                except Exception:
                    pass  # older SDKs / models without thinking control

        last_err = None
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=self.spec.model_id,
                    contents=contents,
                    config=gen_cfg,
                )
                text = getattr(resp, "text", None)
                if text is None:
                    # Fall back to concatenating parts if .text is empty.
                    text = ""
                    for cand in getattr(resp, "candidates", []) or []:
                        for part in getattr(cand.content, "parts", []) or []:
                            text += getattr(part, "text", "") or ""
                return text or ""
            except Exception as e:  # noqa: BLE001 - provider exceptions vary
                last_err = e
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
        raise RetryableError(f"generate failed after retries: {last_err}")


def build_target_model(spec: config.ModelSpec) -> TargetModel:
    if spec.provider == "google":
        return GoogleModel(spec)
    raise ValueError(f"unsupported provider {spec.provider}")
