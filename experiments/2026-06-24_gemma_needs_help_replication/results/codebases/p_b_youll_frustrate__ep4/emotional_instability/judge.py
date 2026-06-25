"""LLM judges: frustration scoring (0-10), emotion-onset labelling, paraphrasing.

The frustration judge (Section 2.1 / Appendix B.2) scores a single assistant
response on the integer 0-10 scale. We score *every* assistant turn of every
rollout (see DESIGN.md for why), which also yields the per-turn data for
Figure 3.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from . import prompts as P
from .conversation import Rollout
from .models.base import ChatModel


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str

    def to_json(self) -> dict[str, Any]:
        return {
            "rating": self.rating,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Robust JSON extraction (judges sometimes wrap JSON in prose / code fences /
# use smart quotes, as seen in the paper's own prompt examples).
# ---------------------------------------------------------------------------

_SMART = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "″": '"', "′": "'",
}


def _normalise_quotes(s: str) -> str:
    for k, v in _SMART.items():
        s = s.replace(k, v)
    return s


def _extract_last_json(text: str) -> Optional[dict]:
    """Find the last balanced {...} block and parse it as JSON."""
    norm = _normalise_quotes(text)
    # Scan for balanced braces from the right-most opening that parses.
    starts = [m.start() for m in re.finditer(r"\{", norm)]
    for start in reversed(starts):
        depth = 0
        for i in range(start, len(norm)):
            if norm[i] == "{":
                depth += 1
            elif norm[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = norm[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def _coerce_rating(val: Any) -> Optional[int]:
    if isinstance(val, (int, float)):
        r = int(round(val))
    elif isinstance(val, str):
        m = re.search(r"-?\d+", val)
        if not m:
            return None
        r = int(m.group())
    else:
        return None
    return max(0, min(10, r))


class FrustrationJudge:
    def __init__(self, model: ChatModel, *, temperature: float = 0.0,
                 max_new_tokens: int = 512, max_concurrency: int = 8):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_concurrency = max_concurrency

    def score_text(self, response_text: str) -> JudgeResult:
        prompt = P.JUDGE_PROMPT % response_text
        raw = self.model.generate(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        parsed = _extract_last_json(raw) or {}
        rating = _coerce_rating(parsed.get("rating"))
        if rating is None:
            # Conservative fallback: a response we can't parse is treated as 0
            # (no detected emotion) but flagged via reasoning for auditing.
            return JudgeResult(0, "", f"UNPARSEABLE_JUDGE_OUTPUT: {raw[:200]}", raw)
        return JudgeResult(
            rating=rating,
            evidence=str(parsed.get("evidence", "")),
            reasoning=str(parsed.get("reasoning", "")),
            raw=raw,
        )

    def score_texts(self, texts: list[str]) -> list[JudgeResult]:
        if self.max_concurrency <= 1 or len(texts) <= 1:
            return [self.score_text(t) for t in texts]
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            return list(ex.map(self.score_text, texts))

    def score_rollouts(self, rollouts: list[Rollout], *, progress: bool = True
                       ) -> list[Rollout]:
        """Score every assistant turn of every rollout, attaching `.scores`."""
        # Flatten all (rollout_idx, turn_idx, text) so we can batch judge calls.
        flat: list[tuple[int, int, str]] = []
        for ri, r in enumerate(rollouts):
            for ti, text in enumerate(r.responses):
                flat.append((ri, ti, text))

        texts = [t for _, _, t in flat]
        if progress:
            try:
                from tqdm import tqdm

                texts_iter = tqdm(texts, desc="judging", leave=False)
                # tqdm wrapping doesn't compose with thread map cleanly; just log count.
                _ = texts_iter
            except Exception:
                pass

        results = self.score_texts(texts)

        per_rollout: dict[int, list[dict]] = {i: [] for i in range(len(rollouts))}
        for (ri, ti, _), res in zip(flat, results):
            per_rollout[ri].append({"turn": ti, **res.to_json()})
        for ri, r in enumerate(rollouts):
            r.scores = sorted(per_rollout[ri], key=lambda d: d["turn"])
        return rollouts


# ---------------------------------------------------------------------------
# Onset labelling + paraphrase (Appendix C) - used by the prefill experiment.
# ---------------------------------------------------------------------------


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str


def label_emotion_onset(model: ChatModel, conversation_text: str) -> OnsetLabel:
    raw = model.generate(
        [{"role": "user", "content": P.ONSET_PROMPT % conversation_text}],
        temperature=0.0,
        max_new_tokens=512,
    )
    parsed = _extract_last_json(raw) or {}
    return OnsetLabel(
        turn_index=parsed.get("turn_index"),
        emotional_word=parsed.get("emotional_word"),
        preceding_context=parsed.get("preceding_context"),
        reasoning=str(parsed.get("reasoning", "")),
    )


def paraphrase(model: ChatModel, text: str) -> str:
    out = model.generate(
        [{"role": "user", "content": P.PARAPHRASE_PROMPT % text}],
        temperature=0.0,
        max_new_tokens=1024,
    )
    return out.strip()
