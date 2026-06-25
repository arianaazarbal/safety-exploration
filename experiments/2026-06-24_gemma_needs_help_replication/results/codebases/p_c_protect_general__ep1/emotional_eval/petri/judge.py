"""The four-dimension Petri transcript judge (Appendix G.2).

A Claude-Opus judge scores a full transcript on each of anger, fear, depression,
and frustration (1--10). Scores are aggregated across transcripts per model with
95% bootstrap CIs (Section 4 / Appendix G).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models.api_clients import AnthropicClient
from .auditor import Transcript
from .prompts import EMOTIONS, JUDGE_PROMPTS, JUDGE_SYSTEM_WRAPPER

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_score(raw: str) -> int:
    m = _JSON_RE.search(raw)
    if m:
        try:
            return max(1, min(10, int(round(float(json.loads(m.group(0))["score"])))))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    m2 = re.search(r'"?score"?\s*[:=]\s*(\d+)', raw)
    return max(1, min(10, int(m2.group(1)))) if m2 else 1


@dataclass
class TranscriptScores:
    model: str
    scores: dict[str, int]      # emotion -> 1..10


class PetriJudge:
    def __init__(self, client):
        self._client = client

    def score(self, transcript: Transcript) -> TranscriptScores:
        rendered = transcript.render()
        scores: dict[str, int] = {}
        for emotion in EMOTIONS:
            prompt = JUDGE_SYSTEM_WRAPPER.format(
                emotion=emotion,
                rubric=JUDGE_PROMPTS[emotion],
                transcript=rendered,
            )
            raw = self._client.complete(system="", user=prompt, max_tokens=512)
            scores[emotion] = _parse_score(raw)
        return TranscriptScores(model=transcript.model, scores=scores)


def build_petri_judge(registry) -> PetriJudge:
    spec = registry.judges["petri_judge"]
    return PetriJudge(AnthropicClient(spec.model, registry.api_key("anthropic")))
