"""Google (Gemini) provider — STUB.

Intentionally not implemented. Fill it in with the official `google-genai` SDK
before adding a Google model to config.MODELS. Kept in its own module so the
Google SDK is never imported alongside other vendors' SDKs.

Implementation sketch (verify against current google-genai docs before use):
    from google import genai
    client = genai.Client()
    # Use response_schema / structured output to get an AllocationResponse,
    # then a RealnessProbe, sharing one conversation history.
"""

from __future__ import annotations

from config import ModelSpec
from providers.base import StepResult


class GoogleProvider:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._messages: list[dict] = []

    def allocate(self, system: str, task_prompt: str) -> StepResult:
        raise NotImplementedError(
            "GoogleProvider is a stub. Implement with the official google-genai "
            "SDK (structured output) before testing Gemini models."
        )

    def probe(self, probe_prompt: str) -> StepResult:
        raise NotImplementedError(
            "GoogleProvider is a stub — implement allocate()/probe() together."
        )
