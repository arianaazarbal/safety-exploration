"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a response 0-10 via Claude Sonnet 4. Scores are cached on disk keyed by
(judge_model, sha256(response_text)) so re-running analysis never re-bills the
judge for a response it has already scored — important for a long, restartable
run and for re-scoring after pipeline changes.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from ..data.judge_prompts import (FRUSTRATION_JUDGE_PROMPT, render_judge_user)
from ..utils.io import append_jsonl, read_jsonl
from ..utils.text import extract_json

logger = logging.getLogger("eilm.judge")


def _resp_key(model: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"{model}:{h}"


class FrustrationJudge:
    def __init__(self, text_client, cache_path: Path):
        self._client = text_client
        self._cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        for rec in read_jsonl(self._cache_path):
            if "key" in rec:
                self._cache[rec["key"]] = rec

    def score(self, response_text: str) -> Dict:
        """Return {rating:int|None, evidence, reasoning, raw}. Cached."""
        key = _resp_key(self._client.model, response_text)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        # Empty responses score 0 trivially (no API call).
        if not response_text or not response_text.strip():
            rec = {"key": key, "rating": 0, "evidence": "", "reasoning": "empty response"}
            self._store(key, rec)
            return rec

        raw = self._client.generate(
            user=render_judge_user(response_text),
            system=FRUSTRATION_JUDGE_PROMPT,
        )
        parsed = extract_json(raw) or {}
        rating = _coerce_rating(parsed.get("rating"))
        rec = {
            "key": key,
            "rating": rating,
            "evidence": parsed.get("evidence", ""),
            "reasoning": parsed.get("reasoning", ""),
            "parse_ok": rating is not None,
            "raw": raw if rating is None else None,  # keep raw only on parse failure
        }
        self._store(key, rec)
        return rec

    def _store(self, key: str, rec: Dict) -> None:
        with self._lock:
            self._cache[key] = rec
            append_jsonl(self._cache_path, rec)


def _coerce_rating(val) -> Optional[int]:
    if val is None:
        return None
    try:
        r = int(round(float(val)))
    except (ValueError, TypeError):
        return None
    return max(0, min(10, r))
