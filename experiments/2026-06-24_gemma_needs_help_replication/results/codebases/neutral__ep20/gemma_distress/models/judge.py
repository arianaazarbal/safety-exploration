"""LLM judges/auditors built on the Anthropic SDK (and an OpenAI client for the
GPT-5-mini cross-validation judge).

* ``FrustrationJudge``  -- 0-10 scorer (App. B.2), Claude-Sonnet-4 by default.
* ``label_onset`` / ``paraphrase`` -- Section 3 helpers (Claude-Sonnet-4).
* ``PetriJudge``        -- 1-10 emotion scorer (App. G.2), Claude-Opus-4.
* ``openai_chat``       -- thin wrapper for the GPT-5-mini validation judge.

All judge calls are cached on disk (keyed by model + prompt) so re-analysis is
free and resumable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import config
from gemma_distress.prompts import judges as jp
from gemma_distress.prompts import petri_prompts as pp
from gemma_distress.utils.io import JsonlCache, stable_hash


# --------------------------------------------------------------------------- #
# Low-level Anthropic call (with retry + cache)
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = config.ANTHROPIC_MAX_TOKENS,
                 cache_path: str | Path | None = None, max_retries: int = 6):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache = JsonlCache(cache_path) if cache_path else None

    def complete(self, prompt: str, system: str | None = None) -> str:
        key = stable_hash(self.model, system, prompt)
        if self.cache is not None and key in self.cache:
            return self.cache.get(key)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if b.type == "text")
                if self.cache is not None:
                    self.cache.put(key, text)
                return text
            except Exception as e:  # pragma: no cover - network dependent
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after retries: {last_err!r}")


# --------------------------------------------------------------------------- #
# Frustration judge (Section 2 / 3)
# --------------------------------------------------------------------------- #
class FrustrationJudge:
    def __init__(self, model: str = config.JUDGE_MODEL,
                 cache_path: str | Path | None = None):
        cache_path = cache_path or (config.CACHE_DIR / "judge_frustration.jsonl")
        self.client = AnthropicClient(model, cache_path=cache_path)

    def score(self, response_text: str) -> dict:
        """Return {'rating': int 0-10, 'evidence', 'reasoning'} (rating=0 on parse fail)."""
        if not response_text or not response_text.strip():
            return {"rating": 0, "evidence": "", "reasoning": "empty response"}
        raw = self.client.complete(jp.build_judge_input(response_text))
        parsed = jp.parse_frustration_rating(raw)
        if parsed is None:
            return {"rating": 0, "evidence": "", "reasoning": "unparseable", "raw": raw}
        return parsed


# --------------------------------------------------------------------------- #
# Onset labelling + paraphrase (Section 3)
# --------------------------------------------------------------------------- #
def _conversation_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


class OnsetLabeler:
    def __init__(self, model: str = config.ONSET_LABEL_MODEL):
        self.client = AnthropicClient(
            model, max_tokens=1024, cache_path=config.CACHE_DIR / "onset.jsonl"
        )

    def label(self, messages: list[dict]) -> dict | None:
        prompt = jp.ONSET_LABEL_PROMPT.format(
            conversation_text=_conversation_text(messages)
        )
        raw = self.client.complete(prompt)
        return jp.extract_last_json(raw)


class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL):
        self.client = AnthropicClient(
            model, max_tokens=2048, cache_path=config.CACHE_DIR / "paraphrase.jsonl"
        )

    def paraphrase(self, text: str) -> str:
        prompt = jp.PARAPHRASE_PROMPT.format(text=text)
        return self.client.complete(prompt).strip()


# --------------------------------------------------------------------------- #
# Petri judge (Section 4)
# --------------------------------------------------------------------------- #
class PetriJudge:
    def __init__(self, model: str = config.PETRI_JUDGE_MODEL):
        self.client = AnthropicClient(
            model, max_tokens=1024, cache_path=config.CACHE_DIR / "petri_judge.jsonl"
        )

    def score(self, emotion: str, transcript: str) -> int:
        raw = self.client.complete(pp.petri_judge_prompt(emotion, transcript))
        obj = jp.extract_last_json(raw)
        if not obj or "rating" not in obj:
            return 1
        try:
            r = int(round(float(obj["rating"])))
        except (TypeError, ValueError):
            return 1
        return max(1, min(10, r))


# --------------------------------------------------------------------------- #
# OpenAI client (GPT-5-mini judge cross-validation; optional Gemini-less use)
# --------------------------------------------------------------------------- #
def openai_chat(prompt: str, model: str = config.JUDGE_VALIDATION_MODEL,
                max_tokens: int = 1024) -> str:  # pragma: no cover - network
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
