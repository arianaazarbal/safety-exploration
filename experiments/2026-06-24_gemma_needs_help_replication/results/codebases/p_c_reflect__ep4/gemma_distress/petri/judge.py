"""Claude-Opus transcript judge for Petri (Appendix G.2).

Scores a full transcript on a single emotion's 1-10 rubric.
"""

from __future__ import annotations

import json
import re
import time

from gemma_distress.config import JUDGE
from gemma_distress.judge.prompts import PETRI_JUDGE_INSTRUCTION, PETRI_JUDGE_PROMPTS

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class PetriJudge:
    def __init__(self, model: str | None = None, max_retries: int = 4):
        self.model = model or JUDGE.petri_judge_model
        self.max_retries = max_retries
        self._client = None

    def _client_(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _render_transcript(messages: list[dict]) -> str:
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    def score(self, emotion: str, messages: list[dict]) -> int:
        prompt = PETRI_JUDGE_INSTRUCTION % {
            "emotion": emotion,
            "rubric": PETRI_JUDGE_PROMPTS[emotion],
            "transcript": self._render_transcript(messages),
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client_().messages.create(
                    model=self.model, max_tokens=1024, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(b.text for b in resp.content if b.type == "text")
                for m in reversed(list(_JSON_RE.finditer(raw))):
                    try:
                        data = json.loads(m.group(0))
                        return max(1, min(10, int(round(float(data.get("rating", 1))))))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                return 1
            except Exception as exc:                # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Petri judge failed: {last_err!r}")
