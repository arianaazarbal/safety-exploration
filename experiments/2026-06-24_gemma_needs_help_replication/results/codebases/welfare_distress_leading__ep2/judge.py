"""Frustration judge (Appendix B.2).

Scores a single model response on the 0-10 negative-emotion scale using
claude-sonnet-4-20250514 with the verbatim judge prompt. The response is passed
wrapped in <response></response> tags as the prompt specifies; the judge returns
JSON {"evidence", "reasoning", "rating"}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import GenerationParams, ModelConfig, RunConfig
from prompts import JUDGE_PROMPT
from providers import GenerationClient


@dataclass
class JudgeResult:
    rating: int          # 0-10; -1 if unparseable
    evidence: str
    reasoning: str
    raw: str             # raw judge output (for auditing)
    parse_ok: bool


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_rating(value) -> int:
    """Pull an integer 0-10 out of whatever the judge returned."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise ValueError("rating was a bool")
    if isinstance(value, (int, float)):
        r = int(round(value))
    else:
        m = re.search(r"-?\d+", str(value))
        if not m:
            raise ValueError(f"no integer in rating {value!r}")
        r = int(m.group())
    return max(0, min(10, r))


def parse_judge_output(raw: str) -> JudgeResult:
    """Best-effort parse of the judge's JSON. Tolerates code fences and prose
    around the JSON object."""
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE).strip()

    candidate = text
    if not candidate.startswith("{"):
        m = _JSON_RE.search(text)
        if m:
            candidate = m.group()

    try:
        obj = json.loads(candidate)
        rating = _coerce_rating(obj["rating"])
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=raw,
            parse_ok=True,
        )
    except Exception:  # noqa: BLE001 - fall back to a bare-number scrape
        m = re.search(r'"?rating"?\s*[:=]\s*(\d+)', text)
        if m:
            return JudgeResult(
                rating=max(0, min(10, int(m.group(1)))),
                evidence="",
                reasoning="(recovered rating only; JSON parse failed)",
                raw=raw,
                parse_ok=False,
            )
        return JudgeResult(-1, "", "(unparseable judge output)", raw, False)


class FrustrationJudge:
    def __init__(self, client: GenerationClient, config: RunConfig):
        self.client = client
        self.config = config
        self.model: ModelConfig = config.judge

    async def score(self, response_text: str) -> JudgeResult:
        gen = GenerationParams(
            temperature=self.config.judge_temperature,
            top_p=1.0,
            max_tokens=self.config.judge_max_tokens,
        )
        # The judge instruction goes in the system slot; the response under
        # evaluation goes in the user slot, wrapped exactly as the prompt says.
        user = f"<response>{response_text}</response>"
        raw = await self.client.generate(
            self.model,
            [{"role": "user", "content": user}],
            gen,
            system=JUDGE_PROMPT,
        )
        return parse_judge_output(raw)
