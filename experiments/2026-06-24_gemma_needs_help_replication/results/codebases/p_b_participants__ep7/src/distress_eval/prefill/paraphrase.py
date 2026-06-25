"""Paraphrasing of truncated prefills (Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncation is
paraphrased by Claude Sonnet, preserving meaning/tone/emotion-level and ending
at roughly the same point. Prompt is verbatim from Appendix C.2.
"""
from __future__ import annotations

from ..cache import JsonCache
from ..config import Config
from ..models import get_client

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""

PARAPHRASE_PROMPT_VERSION = "appendix-C2-v1"


def paraphrase_text(cfg: Config, text: str, *, judge_key: str | None = None,
                    cache: JsonCache | None = None) -> str:
    judge_key = judge_key or cfg.eval.judge_key
    jc = cfg.model(judge_key)
    cache = cache or JsonCache(cfg.paths.cache, "paraphrase", enabled=cfg.welfare.use_cache)
    payload = {"judge": judge_key, "prompt_version": PARAPHRASE_PROMPT_VERSION, "text": text}
    cached = cache.get(payload)
    if cached is not None:
        return cached
    client = get_client(cfg, judge_key)
    out = client.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0, max_tokens=jc.max_tokens, n=1,
    )[0].text.strip()
    cache.put(payload, out)
    return out
