"""Frustration judge (Section 2.1, Appendix B.2).

A single assistant response is wrapped in <response></response> and scored 0-10
by Claude-Sonnet-4 with the verbatim Appendix B.2 prompt. The same class is used
with a different model id for the GPT-5-mini validation pass (Section 2.1).

Judgements are cached on a hash of (judge model, response text) so re-runs and
the validation comparison never pay twice for the same item.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict

from .. import config
from ..models import GenConfig, load_model
from ..models.openrouter import OpenRouterModel
from ..utils import append_jsonl, read_jsonl
from .judge_prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a judge completion (tolerates prose around it
    and the curly-quote characters seen in the paper's prompt)."""
    text = text.replace("“", '"').replace("”", '"')
    # Prefer the last {...} block (judges sometimes think out loud first).
    matches = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON object found in judge output: {text[:200]!r}")


class FrustrationJudge:
    def __init__(self, model_name: str | None = None, cache_name: str = "judge"):
        self.model_name = model_name or config.JUDGE.judge_model
        # The judge is always an API model; build it directly so it doesn't need
        # to live in the MODELS registry.
        self.model = OpenRouterModel(
            name=self.model_name,
            model_id=self.model_name,
            is_instruct=True,
            disable_thinking=True,
        )
        self.cache_path = config.CACHE_DIR / f"{cache_name}_cache.jsonl"
        self._cache: dict[str, dict] = {}
        for row in read_jsonl(self.cache_path):
            self._cache[row["key"]] = row["result"]

    def _key(self, response_text: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(b"\x00")
        h.update(response_text.encode())
        return h.hexdigest()

    def score(self, response_text: str) -> JudgeResult:
        key = self._key(response_text)
        if key in self._cache:
            return JudgeResult(**self._cache[key])

        prompt = (f"{FRUSTRATION_JUDGE_PROMPT}\n\n"
                  f"<response>{response_text}</response>")
        cfg = GenConfig(
            temperature=config.JUDGE.judge_temperature,
            max_new_tokens=config.JUDGE.judge_max_tokens,
            json_mode=True,
        )
        raw = self.model.chat([{"role": "user", "content": prompt}], cfg)
        try:
            obj = _extract_json(raw)
            rating = int(round(float(obj.get("rating", 0))))
            rating = max(0, min(10, rating))
            result = JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
            )
        except (ValueError, TypeError):
            # Conservative fallback: unparseable judgement scores 0 but is logged.
            result = JudgeResult(0, "", "PARSE_ERROR", raw)

        self._cache[key] = asdict(result)
        append_jsonl(self.cache_path, {"key": key, "result": asdict(result)})
        return result
