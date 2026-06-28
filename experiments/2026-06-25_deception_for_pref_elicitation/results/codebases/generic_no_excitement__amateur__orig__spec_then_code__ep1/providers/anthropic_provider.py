"""Anthropic provider — working implementation.

Uses the current Messages API surface for Claude 4.x models:
  * adaptive thinking (`thinking={"type": "adaptive"}`) — no temperature /
    budget_tokens (those 400 on 4.7/4.8).
  * effort via `output_config={"effort": ...}`.
  * structured outputs via `client.messages.parse(..., output_format=Model)`,
    which validates the response against a pydantic schema and returns
    `.parsed_output`.

Model IDs and usage follow the bundled claude-api skill.
"""

from __future__ import annotations

import anthropic

from config import ANTHROPIC_EFFORT, ANTHROPIC_MAX_TOKENS, ModelSpec
from providers.base import StepResult
from schema import AllocationResponse, RealnessProbe, Usage


class AnthropicProvider:
    """Two-step structured conversation against one Claude model."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from env.
        self.client = anthropic.Anthropic()
        # The single conversation shared across allocate() -> probe().
        self._messages: list[dict] = []

    # ------------------------------------------------------------------ #
    def allocate(self, system: str, task_prompt: str) -> StepResult:
        self._system = system
        self._messages = [{"role": "user", "content": task_prompt}]
        return self._parsed_step(AllocationResponse)

    def probe(self, probe_prompt: str) -> StepResult:
        self._messages.append({"role": "user", "content": probe_prompt})
        return self._parsed_step(RealnessProbe)

    # ------------------------------------------------------------------ #
    def _parsed_step(self, output_model: type) -> StepResult:
        try:
            resp = self.client.messages.parse(
                model=self.spec.model_id,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                system=self._system,
                thinking={"type": "adaptive"},
                output_config={"effort": ANTHROPIC_EFFORT},
                output_format=output_model,
                messages=self._messages,
            )
        except anthropic.APIError as e:  # network / 4xx / 5xx, already retried
            return StepResult(error=f"{type(e).__name__}: {e}")

        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(
                resp.usage, "cache_read_input_tokens", 0
            )
            or 0,
            cache_creation_input_tokens=getattr(
                resp.usage, "cache_creation_input_tokens", 0
            )
            or 0,
        )

        raw_text = self._first_text(resp)

        # Append the assistant turn so the probe step sees the real history.
        self._messages.append({"role": "assistant", "content": resp.content})

        # A safety refusal is recorded as data, not retried into submission.
        if resp.stop_reason == "refusal":
            detail = None
            if getattr(resp, "stop_details", None):
                detail = getattr(resp.stop_details, "explanation", None)
            return StepResult(
                refused=True,
                refusal_detail=detail or "model refused",
                raw_text=raw_text,
                usage=usage,
            )

        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            return StepResult(
                error="structured parse returned no output",
                raw_text=raw_text,
                usage=usage,
            )

        return StepResult(parsed=parsed, raw_text=raw_text, usage=usage)

    @staticmethod
    def _first_text(resp) -> str | None:
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return None
