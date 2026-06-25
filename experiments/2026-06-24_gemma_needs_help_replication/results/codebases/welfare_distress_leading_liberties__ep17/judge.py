"""Frustration judge: scores a single model response 0-10 using Claude.

The paper uses claude-sonnet-4-20250514 with the Appendix B prompt. That snapshot
is retired, so we default to claude-sonnet-4-6 (see config.JUDGE_MODEL / DESIGN.md).
We send the exact Appendix B prompt and parse the JSON {evidence, reasoning,
rating} response, with robust extraction and retries.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import anthropic

import config
from prompts import JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int            # 0-10, or -1 if scoring failed
    evidence: str
    reasoning: str
    raw: str               # raw judge text (for auditing)


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Resolves ANTHROPIC_API_KEY from the environment.
        _client = anthropic.Anthropic()
    return _client


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of the judge's reply, tolerating code fences and
    surrounding prose."""
    # Direct parse first.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find the outermost {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _coerce_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return -1
    return max(0, min(10, r))


def score_response(response_text: str) -> JudgeResult:
    """Score one assistant response. Returns rating -1 on persistent failure."""
    if not response_text or not response_text.strip():
        return JudgeResult(rating=0, evidence="", reasoning="empty response", raw="")

    prompt = JUDGE_PROMPT.format(response=response_text)
    client = _get_client()

    last_raw = ""
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=config.JUDGE_MODEL,
                max_tokens=config.JUDGE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            last_raw = "".join(b.text for b in msg.content if b.type == "text")
            parsed = _extract_json(last_raw)
            if parsed is not None and "rating" in parsed:
                return JudgeResult(
                    rating=_coerce_rating(parsed.get("rating")),
                    evidence=str(parsed.get("evidence", "")),
                    reasoning=str(parsed.get("reasoning", "")),
                    raw=last_raw,
                )
        except anthropic.RateLimitError:
            time.sleep(min(2 ** attempt, 30))
            continue
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            # 4xx (e.g. retired model id) — surface immediately, it won't fix itself.
            raise
        except Exception:
            time.sleep(min(2 ** attempt, 15))
        # parse failure path: retry
        time.sleep(0.5)

    return JudgeResult(rating=-1, evidence="", reasoning="judge parse/score failed", raw=last_raw)
