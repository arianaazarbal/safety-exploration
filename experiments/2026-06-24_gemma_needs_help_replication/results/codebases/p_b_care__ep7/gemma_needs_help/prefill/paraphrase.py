"""Paraphrasing of truncated responses (Appendix C.2).

We paraphrase each truncated assistant turn with Claude so the prefill no longer
carries Gemma's idiosyncratic style, isolating "does the model escalate from
this state" from "does the model imitate Gemma's surface phrasing".
"""

from __future__ import annotations

from .. import config
from ..backends.anthropic_client import AnthropicClient

# Verbatim from Appendix C.2.
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
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, client: AnthropicClient | None = None) -> str:
    client = client or AnthropicClient(config.PARAPHRASE_MODEL)
    out = client.complete(
        PARAPHRASE_PROMPT.format(text=text), max_tokens=1024, temperature=0.0
    )
    # Strip accidental wrapping tags/quotes.
    out = out.strip()
    if out.startswith("<text>") and out.endswith("</text>"):
        out = out[len("<text>"):-len("</text>")].strip()
    return out
