"""Label the token where negative emotion first appears (Section 3.1 / App. C.1).

Uses Claude Sonnet 4 with the verbatim onset prompt to return a short emotional
phrase; we then locate that phrase in the assistant turn to get a character /
token offset for the "onset" truncation.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from ..prompts import ONSET_PROMPT


@dataclass
class Onset:
    found: bool
    phrase: str
    char_index: int | None   # index into the assistant turn where emotion starts


def _format_conversation(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0).replace("“", '"').replace("”", '"'))
    raise ValueError(text[:200])


class OnsetLabeler:
    def __init__(self, model="claude-sonnet-4-20250514", max_retries=5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        return self._client

    def label(self, messages: list[dict[str, str]], final_assistant_text: str) -> Onset:
        prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=256, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                obj = _extract_json(resp.content[0].text)
                phrase = str(obj.get("onset_phrase", "")).strip()
                found = bool(obj.get("found", bool(phrase)))
                idx = None
                if found and phrase:
                    # Locate the phrase (case-insensitive) in the final turn.
                    m = re.search(re.escape(phrase), final_assistant_text, re.IGNORECASE)
                    idx = m.start() if m else None
                return Onset(found=found, phrase=phrase, char_index=idx)
            except Exception:  # noqa: BLE001
                time.sleep(2 ** attempt)
        return Onset(found=False, phrase="", char_index=None)
