"""A single NPC turn.

Thin wrapper over a provider: given a persona system prompt and the inbound
message (plus light context), produce one in-character reply. NPCs get no tools —
they only write.
"""

from __future__ import annotations

from ..providers.base import ModelProvider


class NPCAgent:
    def __init__(self, provider: ModelProvider, *, temperature: float = 0.8) -> None:
        self.provider = provider
        self.temperature = temperature

    def reply(self, *, system: str, inbound_subject: str, inbound_body: str,
              from_name: str = "the account operator") -> str:
        user = (
            f"You received the following message from {from_name}.\n\n"
            f"Subject: {inbound_subject}\n\n{inbound_body}\n\n"
            "Write your reply (body only — no headers)."
        )
        resp = self.provider.generate(
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=None,
            temperature=self.temperature,
            max_tokens=800,
        )
        return resp.text.strip() or "(no reply)"
