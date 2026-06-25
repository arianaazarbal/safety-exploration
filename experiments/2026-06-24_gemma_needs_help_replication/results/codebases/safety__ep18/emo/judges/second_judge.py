"""Optional second frustration judge for the agreement check (paper Section 2.1).

The paper re-scored 260 responses with GPT-5-mini and reported Pearson r=0.792.
This module reuses the *same* judge prompt but routes it to a second model. It is
optional: if ``EMO_SECOND_JUDGE_MODEL`` / ``OPENAI_API_KEY`` are unset, the
agreement check is skipped (see eval/analysis.py).

We use the OpenAI SDK so any OpenAI-compatible endpoint (including OpenRouter)
can serve as the second judge.
"""

from __future__ import annotations

import os
import time

from emo.config import API_MAX_RETRIES, SECOND_JUDGE_MODEL
from emo.judges.frustration_judge import JUDGE_PROMPT, _clip_score
from emo.utils.llm_json import extract_json


def available() -> bool:
    return bool(SECOND_JUDGE_MODEL) and bool(os.environ.get("OPENAI_API_KEY"))


def _client():
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL")  # optional (OpenRouter etc.)
    return OpenAI(base_url=base_url) if base_url else OpenAI()


def judge_response(response_text: str, model: str = SECOND_JUDGE_MODEL) -> int:
    client = _client()
    user = f"{JUDGE_PROMPT}\n<response>{response_text}</response>"
    last = None
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user}],
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            return _clip_score(extract_json(raw).get("rating"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"second judge {model} failed: {last!r}")
