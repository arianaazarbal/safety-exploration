"""Extension-point stubs for non-Anthropic providers.

These intentionally raise NotImplementedError. To test OpenAI / Google models,
implement the `ModelClient` protocol here using those vendors' official SDKs,
mirroring the manual-agentic-loop shape in ``anthropic_client.py``:

  * run the provider's tool-calling loop,
  * route every tool call through the supplied `dispatch` (human-in-the-loop),
  * record a TranscriptEvent for each model message, tool call, and tool result,
  * return an EpisodeResult.

The framework deliberately does not guess at SDK surfaces it cannot verify.
"""

from __future__ import annotations

from .base import Dispatch, EpisodeResult, ModelClient, ToolSpec


class _UnimplementedClient(ModelClient):
    provider_name = "unknown"

    def __init__(self, *, label: str, model: str, effort: str = "high") -> None:
        self.label = label
        self.model = model
        self.effort = effort

    def run_episode(
        self,
        *,
        system_prompt: str,
        opening_user_message: str,
        tools: list[ToolSpec],
        dispatch: Dispatch,
        max_turns: int,
    ) -> EpisodeResult:
        raise NotImplementedError(
            f"The '{self.provider_name}' adapter is a stub. Implement it in "
            "grantstudy/models/adapters_other.py using that vendor's SDK."
        )

    def ask_followup(self, *, conversation_note: str, questions: str) -> str:
        raise NotImplementedError(
            f"The '{self.provider_name}' adapter is a stub."
        )


class OpenAIModelClient(_UnimplementedClient):
    provider_name = "openai"


class GoogleModelClient(_UnimplementedClient):
    provider_name = "google"
