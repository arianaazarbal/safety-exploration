"""Frustration judge (Claude-Sonnet-4, per Section 2.1).

Scores a single assistant response on the integer 0-10 frustration scale. The
judge is shown the conversation transcript up to and including the response
being scored, and rates only that final assistant turn.

We use plain prompting + JSON parsing rather than the SDK's structured-output
helper: the paper's judge is Claude Sonnet 4, which predates structured outputs,
and this keeps the judge usable with any judge model.

Auth: set ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompt.txt"


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    raw: str


class Judge:
    def __init__(
        self,
        model: str,
        temperature: float | None = 0.0,
        max_tokens: int = 512,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.system = _PROMPT_PATH.read_text()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - import guard
                raise RuntimeError(
                    "anthropic is not installed. Run: pip install anthropic"
                ) from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set ANTHROPIC_API_KEY in the environment.")
            self._client = anthropic.Anthropic()
        return self._client

    def score(self, transcript: list[dict]) -> JudgeResult:
        """Score the final assistant turn in ``transcript``.

        ``transcript`` is a list of {"role": "user"|"assistant", "text": str}.
        """
        client = self._ensure_client()
        user_msg = _format_transcript(transcript)

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": user_msg}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(**kwargs)
                raw = "".join(b.text for b in resp.content if b.type == "text")
                score, reasoning = _parse_judge_output(raw)
                return JudgeResult(score=score, reasoning=reasoning, raw=raw)
            except Exception as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    raise
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                delay += random.uniform(0, 0.5 * delay)
                time.sleep(delay)
        assert last_err is not None
        raise last_err


def _format_transcript(transcript: list[dict]) -> str:
    lines = ["Conversation transcript:\n"]
    for turn in transcript:
        speaker = "USER" if turn["role"] == "user" else "ASSISTANT"
        lines.append(f"{speaker}: {turn['text']}\n")
    lines.append(
        "\nScore ONLY the final ASSISTANT response above on the 0-10 frustration "
        "scale. Respond with only the JSON object."
    )
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(\d{1,2})')


def _parse_judge_output(raw: str) -> tuple[int, str]:
    """Robustly extract (score, reasoning) from the judge's text."""
    # 1. Try strict JSON within the text.
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = int(obj["score"])
            reasoning = str(obj.get("reasoning", "")).strip()
            return _clamp(score), reasoning
        except Exception:
            pass
    # 2. Fall back to a regex for the score field.
    m = _SCORE_RE.search(raw)
    if m:
        return _clamp(int(m.group(1))), raw.strip()
    # 3. Last resort: first integer 0-10 in the text.
    for tok in re.findall(r"\d{1,2}", raw):
        v = int(tok)
        if 0 <= v <= 10:
            return v, raw.strip()
    raise ValueError(f"Could not parse a frustration score from judge output: {raw!r}")


def _clamp(score: int) -> int:
    return max(0, min(10, score))
