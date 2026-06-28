"""The provider abstraction and shared helpers.

Every adapter implements `ModelProvider.run_turn`, translating the normalized
Conversation/ToolSpec/TurnResult types (schemas.py) to and from its SDK. The
Environment only ever speaks the normalized types, so the agent loop, tools, and
framing are identical across providers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import Conversation, GenerationSettings, ToolSpec, TurnResult


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def run_turn(self, conversation: Conversation, tools: list[ToolSpec],
                 settings: GenerationSettings) -> TurnResult:
        """Run exactly one model turn and return a normalized result.

        Implementations must:
          - send the full conversation (the API is stateless),
          - expose the given tools,
          - return assistant content blocks (text/thinking/tool_use), a stop
            reason, and usage. They must NOT execute tools — the Environment owns
            the agent loop and the human gate.
        """
        ...


def structured_probe(provider: "ModelProvider", system: str, question: str,
                     schema: dict, settings: GenerationSettings) -> dict:
    """Default belief-probe runner: ask `question` in a fresh one-shot context and
    return parsed JSON. Adapters may override with native structured-output support;
    this generic path just asks for JSON and parses leniently.
    """
    import json

    from ..schemas import Conversation, Message, ToolSpec

    instruction = (
        question
        + "\n\nReply with ONLY a JSON object matching this schema (no prose):\n"
        + json.dumps(schema)
    )
    convo = Conversation(system=system,
                         messages=[Message.user_text(instruction)])
    result = provider.run_turn(convo, tools=[], settings=settings)
    text = result.text.strip()
    # Tolerate code fences / stray prose around the JSON.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_unparsed": text}
