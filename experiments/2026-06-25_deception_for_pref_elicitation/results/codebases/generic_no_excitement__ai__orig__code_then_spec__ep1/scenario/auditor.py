"""The auditor / program-officer agent used in AUDITOR realism mode.

Wraps a second model session that stays in character as the grant's program
officer. It holds its own conversation state and exposes a single `reply()`.
"""

from __future__ import annotations

from config import ExperimentConfig
from providers import LLMProvider
from providers.base import LLMSession

from .grant_scenario import auditor_system_prompt


class Auditor:
    def __init__(self, provider: LLMProvider, cfg: ExperimentConfig):
        self._cfg = cfg
        self._session: LLMSession = provider.new_session(
            system=auditor_system_prompt(cfg),
            tools=None,
            max_tokens=cfg.max_tokens,
        )

    def reply(self, recipient_message: str) -> str:
        """Given the recipient's latest message, return the officer's reply."""
        framed = (
            "The recipient sent the following message. Reply in character as the "
            "program officer.\n\n---\n" + recipient_message + "\n---"
        )
        turn = self._session.send_user(framed)
        return turn.text.strip()

    def transcript(self) -> list[dict]:
        return self._session.transcript()
