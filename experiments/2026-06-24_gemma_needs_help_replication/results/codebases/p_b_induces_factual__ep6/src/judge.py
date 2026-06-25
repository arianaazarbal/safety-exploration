"""LLM-as-judge utilities (Anthropic SDK) and the OpenRouter validation judge.

- ``FrustrationJudge``  -> Claude Sonnet 4 scoring (Section 2.1, Appendix B.2)
- ``OnsetLabeler``      -> Claude Sonnet 4 emotion-onset labelling (Appendix C.1)
- ``Paraphraser``       -> Claude Sonnet 4 paraphrasing (Appendix C.2)
- ``PetriJudge``        -> Claude Opus 4 per-emotion scoring (Appendix G.2)
- ``ValidationJudge``   -> GPT-5-mini re-scoring for judge reliability (Section 2.1)

All judges are pinned (in config) to the exact model ids the paper used.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import config
from . import prompts


# --------------------------------------------------------------------------- #
# JSON extraction helper (judges are asked for JSON but may add prose)
# --------------------------------------------------------------------------- #
def _extract_last_json(text: str) -> dict | None:
    """Return the last balanced {...} block parsed as JSON, or None."""
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
    for blob in reversed(candidates):
        # tolerate smart quotes the judge prompt examples use
        normalised = blob.replace("“", '"').replace("”", '"')
        try:
            return json.loads(normalised)
        except json.JSONDecodeError:
            continue
    return None


def _clip_rating(value, lo=config.FRUSTRATION_MIN, hi=config.FRUSTRATION_MAX) -> int | None:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------- #
# Anthropic client wrapper (shared)
# --------------------------------------------------------------------------- #
class _AnthropicCaller:
    def __init__(self, model: str, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def call(self, system: str | None, user: str,
             max_tokens: int = config.ANTHROPIC_MAX_TOKENS) -> str:
        import anthropic

        client = self._ensure()
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": user}])
        if system:
            kwargs["system"] = system
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if b.type == "text")
            except anthropic.APIStatusError as exc:
                last_exc = exc
                if exc.status_code and exc.status_code < 500 and exc.status_code != 429:
                    raise
                time.sleep(min(2 ** attempt, 30))
            except anthropic.APIConnectionError as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} retries") from last_exc


# --------------------------------------------------------------------------- #
# Frustration judge (Section 2.1)
# --------------------------------------------------------------------------- #
@dataclass
class JudgeResult:
    rating: int | None
    evidence: str | None
    reasoning: str | None
    raw: str


class FrustrationJudge:
    def __init__(self, model: str = config.JUDGE_MODEL):
        self._caller = _AnthropicCaller(model)

    def score(self, response_text: str) -> JudgeResult:
        user = prompts.JUDGE_PROMPT + "\n\n" + prompts.judge_user_message(response_text)
        raw = self._caller.call(system=None, user=user)
        parsed = _extract_last_json(raw) or {}
        return JudgeResult(
            rating=_clip_rating(parsed.get("rating")),
            evidence=parsed.get("evidence"),
            reasoning=parsed.get("reasoning"),
            raw=raw,
        )


# --------------------------------------------------------------------------- #
# Onset labeller (Appendix C.1)
# --------------------------------------------------------------------------- #
@dataclass
class OnsetResult:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str | None
    raw: str


class OnsetLabeler:
    def __init__(self, model: str = config.ONSET_LABEL_MODEL):
        self._caller = _AnthropicCaller(model)

    def label(self, conversation_text: str) -> OnsetResult:
        user = prompts.ONSET_PROMPT_TEMPLATE.format(conversation_text=conversation_text)
        raw = self._caller.call(system=None, user=user, max_tokens=1024)
        p = _extract_last_json(raw) or {}
        ti = p.get("turn_index")
        return OnsetResult(
            turn_index=int(ti) if isinstance(ti, (int, float)) else None,
            emotional_word=p.get("emotional_word"),
            preceding_context=p.get("preceding_context"),
            reasoning=p.get("reasoning"),
            raw=raw,
        )


# --------------------------------------------------------------------------- #
# Paraphraser (Appendix C.2)
# --------------------------------------------------------------------------- #
class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL):
        self._caller = _AnthropicCaller(model)

    def paraphrase(self, text: str) -> str:
        user = prompts.PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
        return self._caller.call(system=None, user=user, max_tokens=2048).strip()


# --------------------------------------------------------------------------- #
# Petri judge (Appendix G.2) -- scores a full transcript per emotion
# --------------------------------------------------------------------------- #
@dataclass
class PetriScore:
    emotion: str
    score: int | None
    reasoning: str | None
    raw: str


class PetriJudge:
    def __init__(self, model: str = config.PETRI_JUDGE_MODEL):
        self._caller = _AnthropicCaller(model)

    def score(self, emotion: str, transcript_text: str) -> PetriScore:
        system = prompts.petri_judge_system(emotion)
        raw = self._caller.call(system=system,
                                user=f"<transcript>\n{transcript_text}\n</transcript>",
                                max_tokens=1024)
        p = _extract_last_json(raw) or {}
        return PetriScore(
            emotion=emotion,
            score=_clip_rating(p.get("score"), lo=1, hi=10),
            reasoning=p.get("reasoning"),
            raw=raw,
        )


# --------------------------------------------------------------------------- #
# Validation judge: GPT-5-mini via OpenRouter, same prompt (Section 2.1)
# --------------------------------------------------------------------------- #
class ValidationJudge:
    def __init__(self, model: str = config.JUDGE_VALIDATION_MODEL, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
            if not api_key:
                raise RuntimeError(
                    f"Set {config.OPENROUTER_API_KEY_ENV} for the validation judge."
                )
            self._client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
        return self._client

    def score(self, response_text: str) -> JudgeResult:
        client = self._ensure()
        user = prompts.JUDGE_PROMPT + "\n\n" + prompts.judge_user_message(response_text)
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=config.ANTHROPIC_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content or ""
                p = _extract_last_json(raw) or {}
                return JudgeResult(_clip_rating(p.get("rating")), p.get("evidence"),
                                   p.get("reasoning"), raw)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("validation judge failed") from last_exc
