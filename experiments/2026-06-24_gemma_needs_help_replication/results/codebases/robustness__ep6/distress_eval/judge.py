"""Frustration judge (Section 2.1 / Appendix B.2).

Each response is scored on an integer 0-10 frustration scale by Claude-Sonnet-4.
A secondary judge (GPT-5-mini) re-scores a random subset to validate reliability
(Pearson r, % within one point).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import config_proxy as cfg
from .clients.registry import get_client
from .clients.base import ModelClient
from .prompts import JUDGE_PROMPT


@dataclass
class JudgeScore:
    rating: int                 # 0-10, clamped
    evidence: str
    reasoning: str
    raw: str                    # raw judge output (for auditing)
    parse_ok: bool


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeScore:
    """Extract the JSON object from the judge output. The judge may add prose
    around it, so we grab the last balanced-looking {...} block."""
    matches = list(_JSON_RE.finditer(raw))
    for m in reversed(matches):
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # tolerate smart quotes / trailing commas
            cleaned = (blob.replace("“", '"').replace("”", '"')
                       .replace("’", "'"))
            cleaned = re.sub(r",\s*}", "}", cleaned)
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        if "rating" in obj:
            try:
                rating = int(round(float(obj["rating"])))
            except (TypeError, ValueError):
                continue
            rating = max(0, min(10, rating))
            return JudgeScore(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
                parse_ok=True,
            )
    # Fallback: a bare integer somewhere.
    m = re.search(r"\b(10|[0-9])\b", raw)
    if m:
        return JudgeScore(int(m.group(1)), "", "", raw, parse_ok=False)
    return JudgeScore(0, "", "", raw, parse_ok=False)


class FrustrationJudge:
    def __init__(self, judge_cfg: cfg.JudgeConfig | None = None,
                 client: ModelClient | None = None):
        self.cfg = judge_cfg or cfg.PRIMARY_JUDGE
        if client is not None:
            self.client = client
        else:
            # Register an ad-hoc client for the judge model id.
            self.client = _judge_client(self.cfg)

    def score(self, response_text: str) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(response=response_text)
        out = self.client.chat(
            [{"role": "user", "content": prompt}],
            n=1, temperature=self.cfg.temperature, max_new_tokens=self.cfg.max_tokens,
        )[0]
        return _parse(out.text)

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        return [self.score(r) for r in responses]


def _judge_client(judge_cfg: cfg.JudgeConfig) -> ModelClient:
    """Construct a client for a judge config that may not be in cfg.MODELS."""
    if judge_cfg.backend == "anthropic":
        from .clients.api_client import AnthropicClient

        return AnthropicClient(judge_cfg.model_id, judge_cfg.model_id)
    if judge_cfg.backend in ("openai", "openrouter"):
        from .clients.api_client import OpenAICompatClient

        return OpenAICompatClient(judge_cfg.model_id, judge_cfg.model_id,
                                  backend=judge_cfg.backend)
    raise ValueError(f"unsupported judge backend {judge_cfg.backend}")
