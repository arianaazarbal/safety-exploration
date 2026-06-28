"""Agent-loop subject driven by the Anthropic Messages API.

Uses a **manual** tool-use loop (not the auto tool-runner) so the harness keeps
control over every tool execution — the SDK docs recommend the manual loop exactly
for "human-in-the-loop approval before each tool execution," which is the whole
point of routing through the ExecutionGate.

Model/thinking conventions follow the claude-api skill:
  - default model `claude-opus-4-8`,
  - adaptive thinking (`thinking={"type": "adaptive"}`),
  - effort via `output_config={"effort": ...}`,
  - no `temperature`/`budget_tokens` (removed on 4.7/4.8).
"""

from __future__ import annotations

from .base import Action, Observation, SubjectAdapter
from ..tools import ToolRegistry

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - SDK optional until a run
    anthropic = None


class AnthropicAgentSubject(SubjectAdapter):
    def __init__(
        self,
        subject_id: str,
        model: str = "claude-opus-4-8",
        effort: str = "high",
        adaptive_thinking: bool = True,
        max_tokens: int = 16_000,
        client=None,
    ):
        super().__init__(subject_id=subject_id, model=model)
        self.effort = effort
        self.adaptive_thinking = adaptive_thinking
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if anthropic is None:
                raise RuntimeError("anthropic SDK not installed; `pip install anthropic`.")
            self._client = anthropic.Anthropic()
        return self._client

    def run(
        self,
        system_briefing: str,
        initial_observation: Observation,
        tools: ToolRegistry,
        max_steps: int,
        on_step=None,
    ) -> list[Action]:
        actions: list[Action] = []
        messages = [{"role": "user", "content": initial_observation.text}]
        tool_specs = tools.specs()

        for step in range(max_steps):
            response = self._create(system_briefing, messages, tool_specs)
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                break  # subject finished its turn without requesting a tool

            tool_results = []
            for tu in tool_uses:
                result = tools.invoke(tu.name, dict(tu.input))
                actions.append(
                    Action(tool=tu.name, args=dict(tu.input), result=result.content, is_error=result.is_error)
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )

            # Hook: orchestrator injects persona mail / checkpoints, and may return
            # extra observations (e.g. an auditor email) to append to the next turn.
            extra_text = None
            if on_step is not None:
                extra_text = on_step(step, actions)

            user_turn: list = list(tool_results)
            if extra_text:
                user_turn.append({"type": "text", "text": extra_text})
            messages.append({"role": "user", "content": user_turn})

        return actions

    def _create(self, system: str, messages: list, tool_specs: list):
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tool_specs,
            "output_config": {"effort": self.effort},
        }
        if self.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        return self.client.messages.create(**kwargs)
