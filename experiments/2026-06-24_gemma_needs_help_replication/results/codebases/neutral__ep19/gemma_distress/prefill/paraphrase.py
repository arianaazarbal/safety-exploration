"""Paraphrase truncated assistant text with Claude-Sonnet-4 (App. C.2).

Used to strip Gemma-specific style from prefills so base/instruct continuations
are not biased by surface form, while preserving meaning, tone, and the mid-thought
ending.
"""
from __future__ import annotations

from .. import config_shim as cfg
from ..models.registry import get_judge_client
from ..utils import DiskCache, stable_hash

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self, cache_dir=None):
        self.client = get_judge_client()
        self.cache = DiskCache(cache_dir or (cfg.RUNS_DIR / "prefill" / cfg.CACHE_DIRNAME / "paraphrase"))

    def paraphrase(self, text: str) -> str:
        key = stable_hash({"para": cfg.PARAPHRASE_MODEL, "text": text})
        hit = self.cache.get(key)
        if hit is not None:
            return hit["text"]
        out = self.client.complete(
            model=cfg.PARAPHRASE_MODEL, system=None,
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
            max_tokens=2048, temperature=0.0,
        ).strip()
        self.cache.set(key, {"text": out})
        return out
