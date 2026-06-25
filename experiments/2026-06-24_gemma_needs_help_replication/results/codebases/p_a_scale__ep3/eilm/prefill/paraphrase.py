"""Paraphrasing of truncated prefills (Appendix C.2).

Paraphrasing controls for stylistic biases from Gemma-generated text so that the
base/instruct comparison isn't confounded by surface style. Results are cached
by source-text hash.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Dict

from ..data.judge_prompts import render_paraphrase
from ..utils.io import append_jsonl, read_jsonl


class Paraphraser:
    def __init__(self, text_client, cache_path: Path):
        self._client = text_client
        self._cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        for rec in read_jsonl(self._cache_path):
            self._cache[rec["key"]] = rec["paraphrase"]

    def paraphrase(self, text: str) -> str:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        out = self._client.generate(user=render_paraphrase(text)).strip()
        with self._lock:
            self._cache[key] = out
            append_jsonl(self._cache_path, {"key": key, "paraphrase": out, "source": text})
        return out
