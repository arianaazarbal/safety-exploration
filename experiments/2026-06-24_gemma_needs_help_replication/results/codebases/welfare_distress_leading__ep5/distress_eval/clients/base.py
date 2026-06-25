"""Provider-agnostic chat client interface.

Every backend implements `chat(messages, ...) -> str`. Messages use a neutral
schema so the rest of the codebase never depends on a particular SDK:

    {"role": "system" | "user" | "assistant", "content": "<text>"}

Backends translate this to their own format. System messages are optional and
may be merged/dropped by backends whose models lack a system role (e.g. Gemma).
"""

from __future__ import annotations

from typing import Dict, List, Protocol

Message = Dict[str, str]


class ChatClient(Protocol):
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the assistant's text completion for `messages`."""
        ...


def split_system(messages: List[Message]) -> tuple[str | None, List[Message]]:
    """Pull a leading system message out, returning (system_text, rest).

    Helper for backends that take the system prompt as a separate argument.
    """
    system = None
    rest: List[Message] = []
    for m in messages:
        if m["role"] == "system" and system is None and not rest:
            system = m["content"]
        else:
            rest.append(m)
    return system, rest
