"""Thin Anthropic text-completion helper for onset-labelling and paraphrasing.

Both tasks are simple single-prompt -> text calls against the paper-pinned
Claude-Sonnet snapshot. Threaded + retried like the judge.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from tenacity import retry, stop_after_attempt, wait_exponential


class AnthropicText:
    def __init__(self, model: str, *, max_tokens: int = 1024, max_workers: int = 8):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
    def complete(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def complete_many(self, prompts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.complete, prompts))


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_onset_json(text: str) -> dict | None:
    """Parse the trailing JSON of an onset-labelling response (Appendix C.1)."""
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    matches = list(_JSON_RE.finditer(cleaned))
    for m in reversed(matches):           # the JSON is at the end of the response
        try:
            data = json.loads(m.group(0))
            if "turn_index" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None
