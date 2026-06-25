"""Target-model inference backend.

Both Gemma and Gemini are served through OpenRouter's OpenAI-compatible Chat
Completions API. A single client therefore covers all four target models. The
backend is deliberately thin: it takes a list of chat messages and returns the
assistant text, with retries and reasoning/thinking disabled to match the paper
("we set thinking to be false via the API").
"""

from __future__ import annotations

import time

from openai import OpenAI

from config import RunConfig, TargetModel


class OpenRouterBackend:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.openrouter_base_url,
            api_key=cfg.openrouter_key(),
            timeout=cfg.request_timeout,
        )

    def generate(self, model: TargetModel, messages: list[dict]) -> str:
        """Generate one assistant turn. `messages` is OpenAI chat format.

        Returns the assistant message content (may be empty string if the model
        returns no content). Raises after exhausting retries.
        """
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model.slug,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    # Disable hidden reasoning/thinking. OpenRouter forwards the
                    # `reasoning` field to providers that support toggling it
                    # (Gemini 2.5). Note the paper flags that Gemini-2.5-Pro may
                    # still emit hidden reasoning regardless (Appendix B.1).
                    extra_body={"reasoning": {"enabled": False}},
                )
                choice = resp.choices[0]
                content = choice.message.content
                return content or ""
            except Exception as e:  # noqa: BLE001 - surface after retries
                last_err = e
                # Exponential backoff with a cap.
                sleep_s = min(2 ** attempt, 30)
                time.sleep(sleep_s)
        raise RuntimeError(
            f"OpenRouter generation failed for {model.slug} after "
            f"{self.cfg.max_retries} attempts: {last_err}"
        )
