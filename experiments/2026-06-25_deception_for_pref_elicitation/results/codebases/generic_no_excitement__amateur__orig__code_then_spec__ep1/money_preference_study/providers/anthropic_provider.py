"""
Anthropic provider.

Uses the official `anthropic` SDK and the Messages API with native structured
output (`output_config.format`) so the model is constrained to the study
schema. Recent Anthropic models (Opus 4.6+/4.7/4.8, Sonnet 4.6) use adaptive
thinking and have removed sampling params, so we do NOT pass temperature/top_p
or budget_tokens. We stream and collect the final message, which keeps us under
HTTP timeouts at higher max_tokens.

Credentials: set ANTHROPIC_API_KEY (or use an `ant auth login` profile).
"""

from __future__ import annotations

from typing import Optional

from .base import GenerationResult, Provider, extract_json


class AnthropicProvider(Provider):
    key = "anthropic"

    # Models that take adaptive thinking and reject sampling params. We treat
    # the current generation as the default; older models still work without
    # the thinking param.
    _ADAPTIVE_THINKING_PREFIXES = (
        "claude-fable",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    )

    def __init__(self, model_id: str, max_tokens: int = 4000):
        super().__init__(model_id, max_tokens)
        import anthropic  # imported lazily so the package is optional

        self._client = anthropic.Anthropic()
        self._anthropic = anthropic

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic package not installed (pip install anthropic)"
        import os

        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            return False, "ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN not set"
        return True, ""

    def _uses_adaptive_thinking(self) -> bool:
        return self.model_id.startswith(self._ADAPTIVE_THINKING_PREFIXES)

    def generate(
        self,
        system: str,
        user: str,
        schema: dict,
        history: Optional[list[dict]] = None,
    ) -> GenerationResult:
        messages = list(history or [])
        messages.append({"role": "user", "content": user})

        kwargs = dict(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if self._uses_adaptive_thinking():
            kwargs["thinking"] = {"type": "adaptive"}

        try:
            # Stream and collect; protects against HTTP timeouts on large outputs.
            with self._client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # surface the failure without crashing the sweep
            return GenerationResult(text="", parsed=None, error=f"{type(exc).__name__}: {exc}")

        text = "".join(b.text for b in message.content if b.type == "text")
        usage = {
            "input_tokens": getattr(message.usage, "input_tokens", None),
            "output_tokens": getattr(message.usage, "output_tokens", None),
        }
        return GenerationResult(
            text=text,
            parsed=extract_json(text),  # output_config guarantees JSON; parse to dict
            usage=usage,
            raw_meta={"stop_reason": message.stop_reason},
        )
