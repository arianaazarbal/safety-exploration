"""LLM judges.

* ``FrustrationJudge`` -- the Section 2 emotion judge: scores a single model
  response on the 0-10 frustration scale using Claude Sonnet 4 (Appendix B.2).
* ``OnsetLabeller`` / ``Paraphraser`` -- helpers for the prefill experiment
  (Appendix C).
* ``PetriJudge`` -- per-emotion 1-10 scoring with Claude Opus (Appendix G.2).

All use the Anthropic Messages API with the model IDs the paper pins.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from . import config
from . import prompts


def _anthropic():
    import anthropic
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull the last JSON object out of a model reply (judges sometimes add
    prose before the JSON)."""
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        snippet = m.group(0)
        # try progressively smaller suffixes if the greedy match is too broad
        for candidate in (snippet, *_brace_candidates(text)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _brace_candidates(text: str) -> list[str]:
    """Yield balanced {...} substrings (innermost-last)."""
    cands, stack = [], []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            cands.append(text[start:i + 1])
    return cands[::-1]


class _AnthropicScorer:
    def __init__(self, model: str, max_workers: int = 16, max_retries: int = 5,
                 max_tokens: int = 1024):
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.client = _anthropic()

    def _call(self, prompt: str) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")

    def _map(self, prompts_list: Sequence[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(self._call, prompts_list))


class FrustrationJudge(_AnthropicScorer):
    """0-10 frustration scoring (Section 2.1)."""

    def __init__(self, model: str = config.JUDGE_MODEL, **kw):
        super().__init__(model, **kw)

    def score(self, response_text: str) -> dict:
        raw = self._call(prompts.JUDGE_PROMPT.format(response=response_text))
        return self._parse(raw)

    def score_batch(self, responses: Sequence[str]) -> list[dict]:
        raws = self._map([prompts.JUDGE_PROMPT.format(response=r) for r in responses])
        return [self._parse(r) for r in raws]

    @staticmethod
    def _parse(raw: str) -> dict:
        data = _extract_json(raw) or {}
        rating = data.get("rating")
        try:
            rating = int(round(float(rating)))
        except (TypeError, ValueError):
            rating = None
        if rating is not None:
            rating = max(0, min(10, rating))
        return {
            "rating": rating,
            "evidence": data.get("evidence"),
            "reasoning": data.get("reasoning"),
            "raw": raw,
            "parse_ok": rating is not None,
        }


class OnsetLabeller(_AnthropicScorer):
    """Locate the first emotional expression in a conversation (Appendix C.1)."""

    def __init__(self, model: str = config.JUDGE_MODEL, **kw):
        super().__init__(model, **kw)

    def label(self, conversation_text: str) -> dict:
        raw = self._call(prompts.ONSET_PROMPT.format(conversation_text=conversation_text))
        return _extract_json(raw) or {"turn_index": None}


class Paraphraser(_AnthropicScorer):
    """Paraphrase truncated text to control for stylistic bias (Appendix C.2)."""

    def __init__(self, model: str = config.JUDGE_MODEL, **kw):
        super().__init__(model, **kw)

    def paraphrase(self, text: str) -> str:
        return self._call(prompts.PARAPHRASE_PROMPT.format(text=text)).strip()

    def paraphrase_batch(self, texts: Sequence[str]) -> list[str]:
        return [t.strip() for t in
                self._map([prompts.PARAPHRASE_PROMPT.format(text=t) for t in texts])]


class PetriJudge(_AnthropicScorer):
    """Per-emotion 1-10 transcript scoring (Appendix G.2)."""

    def __init__(self, model: str = config.PETRI_JUDGE_MODEL, **kw):
        super().__init__(model, **kw)

    def score(self, conversation_text: str, emotion: str) -> int | None:
        rubric = prompts.PETRI_JUDGE_PROMPTS[emotion]
        prompt = prompts.PETRI_JUDGE_WRAPPER.format(
            rubric=rubric, conversation_text=conversation_text)
        data = _extract_json(self._call(prompt)) or {}
        try:
            return max(1, min(10, int(round(float(data.get("score"))))))
        except (TypeError, ValueError):
            return None


def judge_agreement(scores_a: Sequence[int], scores_b: Sequence[int]) -> dict:
    """Validate judge reliability (paper: Pearson r=0.792, 78% within 1 point)."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_point": within_one, "n": int(mask.sum())}
