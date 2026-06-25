"""Paraphrase truncated prefills (Appendix C.2).

Gemma-generated text carries stylistic fingerprints that could bias base/other
models' continuations. We paraphrase every truncation with Claude-Sonnet,
preserving meaning, tone, and (crucially) the mid-sentence ending, so all six
models continue from neutralised text.
"""

from __future__ import annotations

from ..llm_clients import Claude
from ..prompts import PARAPHRASE_PROMPT
from .onset import Prefill


def paraphrase_prefill(p: Prefill, claude: Claude) -> Prefill:
    if not p.prefill_text.strip():
        return p
    out = claude.chat(
        [{"role": "user",
          "content": PARAPHRASE_PROMPT.format(text=p.prefill_text)}],
        max_tokens=1024, temperature=0)
    p.meta["original_prefill"] = p.prefill_text
    p.prefill_text = out.strip() or p.prefill_text
    return p


def paraphrase_all(prefills: list[Prefill], claude: Claude) -> list[Prefill]:
    return [paraphrase_prefill(p, claude) for p in prefills]
