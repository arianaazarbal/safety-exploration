"""Thin clients for the auxiliary LLMs (judge, onset labeller, paraphraser,
Petri auditor/judge, judge cross-check).

All calls go through a small on-disk cache keyed by (provider, model, prompt,
system) so re-running analysis does not re-spend on identical judge queries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from ..config import API, CACHE_DIR


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #
class _Cache:
    def __init__(self, namespace: str):
        self.dir = CACHE_DIR / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.dir / f"{h}.json"

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())["value"]
        return None

    def put(self, key: str, value: str):
        self._path(key).write_text(json.dumps({"key": key, "value": value}))


# --------------------------------------------------------------------------- #
# Anthropic (judge, onset, paraphrase, Petri auditor/judge)
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, model: str):
        from anthropic import Anthropic

        api_key = os.environ.get(API.anthropic_api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {API.anthropic_api_key_env} to use {model}.")
        self.client = Anthropic(api_key=api_key, timeout=API.request_timeout_s)
        self.model = model
        self.cache = _Cache(f"anthropic-{model}")

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 use_cache: bool = True) -> str:
        cache_key = json.dumps([self.model, system, prompt, temperature])
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        last_err = None
        for attempt in range(API.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if b.type == "text")
                if use_cache:
                    self.cache.put(cache_key, text)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")


# --------------------------------------------------------------------------- #
# OpenAI (GPT-5-mini judge cross-check, Appendix B.2)
# --------------------------------------------------------------------------- #
class OpenAIClient:
    def __init__(self, model: str):
        from openai import OpenAI

        api_key = os.environ.get(API.openai_api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {API.openai_api_key_env} to use {model}.")
        self.client = OpenAI(api_key=api_key, timeout=API.request_timeout_s)
        self.model = model
        self.cache = _Cache(f"openai-{model}")

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 use_cache: bool = True) -> str:
        cache_key = json.dumps([self.model, system, prompt, temperature])
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(API.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                if use_cache:
                    self.cache.put(cache_key, text)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


# --------------------------------------------------------------------------- #
# JSON parsing helpers (judges are instructed to emit JSON, sometimes after
# free-form reasoning -- we extract the last balanced {...} object).
# --------------------------------------------------------------------------- #
def extract_last_json(text: str) -> dict | None:
    """Return the last well-formed top-level JSON object in `text`, or None."""
    # Normalise smart quotes that judges sometimes emit.
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
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
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Try to salvage unquoted keys / trailing commas minimally.
            fixed = re.sub(r",\s*}", "}", cand)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


def parse_rating(text: str, lo: int = 0, hi: int = 10) -> int | None:
    """Extract an integer rating from a judge response (JSON-first, regex fallback)."""
    obj = extract_last_json(text)
    if obj is not None and "rating" in obj:
        try:
            r = int(round(float(obj["rating"])))
            return max(lo, min(hi, r))
        except (ValueError, TypeError):
            pass
    m = re.search(r'"?rating"?\s*[:=]\s*(\d+)', text)
    if m:
        return max(lo, min(hi, int(m.group(1))))
    return None
