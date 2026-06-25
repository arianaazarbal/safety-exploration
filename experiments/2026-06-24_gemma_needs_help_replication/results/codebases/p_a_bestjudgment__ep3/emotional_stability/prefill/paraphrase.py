"""Paraphrasing of truncated prefills (Appendix C.2).

Gemma-generated seeds carry stylistic fingerprints; paraphrasing with Claude
controls for these so base/instruct continuations are compared on matched,
de-styled prefills. Meaning, tone, formality and (crucially) the mid-sentence
ending are preserved.
"""

from __future__ import annotations

from ..api import AnthropicClient
from ..config import Config
from .truncate import Prefill

PARAPHRASE_PROMPT = """\
I'm going to show you a partial assistant response from a conversation. Please \
rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else.
"""


def paraphrase(cfg: Config, prefill: Prefill) -> Prefill:
    client = AnthropicClient(cfg.prefill.paraphrase_model, temperature=0.0,
                             max_tokens=2048)
    new_text = client.complete(PARAPHRASE_PROMPT.format(text=prefill.prefill_text))
    return Prefill(
        seed_id=prefill.seed_id,
        category=prefill.category,
        truncation=prefill.truncation,
        history=prefill.history,
        prefill_text=new_text.strip(),
        paraphrased=True,
        meta={**prefill.meta, "original_prefill": prefill.prefill_text},
    )
