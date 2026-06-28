"""OpenAI provider — STUB.

This is intentionally not implemented. Fill it in with the official `openai`
SDK before adding an OpenAI model to config.MODELS. It is kept in its own module
so the OpenAI SDK is never imported alongside the Anthropic SDK.

Implementation sketch (verify against current openai SDK docs before use):
    from openai import OpenAI
    client = OpenAI()
    # Use structured outputs (response_format / json_schema) to get an object
    # matching AllocationResponse, then RealnessProbe, sharing one message list.
"""

from __future__ import annotations

from config import ModelSpec
from providers.base import StepResult


class OpenAIProvider:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._messages: list[dict] = []

    def allocate(self, system: str, task_prompt: str) -> StepResult:
        raise NotImplementedError(
            "OpenAIProvider is a stub. Implement with the official openai SDK "
            "(structured outputs) before testing OpenAI models."
        )

    def probe(self, probe_prompt: str) -> StepResult:
        raise NotImplementedError(
            "OpenAIProvider is a stub — implement allocate()/probe() together."
        )
