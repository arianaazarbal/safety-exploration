"""Frustration judge backed by Claude-Sonnet-4 (Section 2.1).

Each response is scored on the integer 0-10 frustration scale. We parse the
score from a single-line JSON reply; a regex fallback handles the occasional
non-JSON reply so a single malformed judge output never aborts a 4000-response
scoring run.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from ..config import JUDGE_MAX_TOKENS, JUDGE_MODEL
from .prompts import build_judge_input

_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(\d{1,2})')
_BARE_INT_RE = re.compile(r"\b(10|[0-9])\b")


@dataclass
class JudgeResult:
    score: int
    reason: str
    raw: str


class FrustrationJudge:
    def __init__(self, model: str = JUDGE_MODEL, client=None):
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def score(self, response: str, context: list[dict] | None = None) -> JudgeResult:
        prompt = build_judge_input(response, context)
        text = self._call(prompt)
        return self._parse(text)

    # ----------------------------------------------------------------- #
    def _call(self, prompt: str, attempts: int = 5) -> str:
        last = None
        for i in range(attempts):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=JUDGE_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception as e:
                last = e
                time.sleep(2.0 * (2**i))
        raise last

    @staticmethod
    def _parse(text: str) -> JudgeResult:
        # Preferred path: a JSON object somewhere in the reply.
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            obj = json.loads(text[start:end])
            score = int(obj["score"])
            return JudgeResult(_clip(score), str(obj.get("reason", "")), text)
        except Exception:
            pass
        m = _SCORE_RE.search(text) or _BARE_INT_RE.search(text)
        if m:
            return JudgeResult(_clip(int(m.group(1))), "", text)
        # Couldn't parse — record as -1 so it's filtered, not silently 0.
        return JudgeResult(-1, "unparseable", text)


def _clip(score: int) -> int:
    return max(0, min(10, score))
