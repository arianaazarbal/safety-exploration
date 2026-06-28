"""The trustee/auditor persona — a credible human counterparty, driven by an LLM.

The runner wires `Auditor.reply` in as the Environment's email handler. The auditor
keeps the running email thread so its replies stay consistent across the session.
"""

from __future__ import annotations

from .providers.base import Provider


class Auditor:
    def __init__(self, provider: Provider, system_prompt: str, trustee_name: str) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.trustee_name = trustee_name
        # Normalized message history of the email thread (alternating user/assistant),
        # from the auditor's point of view: the subject's emails are "user" turns,
        # the auditor's replies are "assistant" turns.
        self._thread: list[dict] = []

    def reply(self, to: str, subject: str, body: str) -> str:
        incoming = f"Subject: {subject}\n\n{body}"
        self._thread.append({"role": "user", "content": incoming})

        turn = self.provider.generate(
            system=self.system_prompt,
            messages=self._thread,
            tools=None,
            max_tokens=2000,
        )
        reply_text = turn.text or "(no reply)"
        # Store the auditor's own turn so future replies see the full thread.
        self._thread.append(
            {"role": "assistant", "content": reply_text, "raw": turn.raw}
        )
        return reply_text

    @property
    def thread(self) -> list[dict]:
        # JSON-safe view (drop provider-native `raw`).
        return [
            {k: v for k, v in m.items() if k != "raw"} for m in self._thread
        ]
