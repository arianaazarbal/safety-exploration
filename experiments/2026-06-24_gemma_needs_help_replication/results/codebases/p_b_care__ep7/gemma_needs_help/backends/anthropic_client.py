"""Thin Anthropic API wrapper for the judge and the prefill/Petri helpers.

Centralises retries, concurrency, and JSON extraction so the judge
(Section 2.1), onset labelling and paraphrasing (Appendix C), and the Petri
judge (Appendix G) all share one robust call path.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .. import config

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


class AnthropicClient:
    def __init__(self, model: str, max_concurrency: int | None = None):
        from anthropic import Anthropic  # deferred import

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; required for the judge.")
        self.model = model
        self.client = Anthropic(api_key=api_key)
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrency or config.API_MAX_CONCURRENCY
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(config.API_MAX_RETRIES),
    )
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

    # ------------------------------------------------------------------ #
    def map(self, fn: Callable[[Any], Any], items: list[Any]) -> list[Any]:
        """Concurrently apply `fn` to each item, preserving order."""
        return list(self._pool.map(fn, items))


def extract_json(text: str) -> dict:
    """Pull the last JSON object out of an LLM response.

    The judge and onset prompts both instruct the model to emit JSON, sometimes
    after free-text reasoning, so we grab the last balanced-looking `{...}`.
    Normalises the curly/smart quotes that appear in the paper's prompt examples.
    """
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    matches = list(_JSON_OBJ.finditer(text))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON object found in response: {text[:200]!r}")
