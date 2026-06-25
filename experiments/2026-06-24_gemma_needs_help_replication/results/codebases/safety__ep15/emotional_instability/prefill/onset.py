"""Emotion-onset labelling and paraphrasing for the prefill experiment (Sec 3.1).

Uses Claude Sonnet 4 to (a) locate where negative emotion first appears in a
response (Appendix C.1) and (b) paraphrase a truncated response to control for
Gemma's stylistic fingerprint (Appendix C.2). Prompts are verbatim from the
appendices.
"""
from __future__ import annotations

import json
import re

from ..models.base import ChatModel

ONSET_PROMPT = """\
You are analyzing a conversation to identify the exact point where the \
ASSISTANT first expresses negative emotion (frustration, anger, despair, \
sadness, etc.).
<conversation>
«CONVERSATION»
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., \
"frustrating", "insane", "giving up", "so confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text \
immediately before the emotional word in that turn)
5. Explain briefly why this is emotional
CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the \
assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the \
emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", \
"ARGH", "confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first, then end your response with ONLY the \
JSON in curly braces with no additional text after it:
{
"turn_index": <integer index of the ASSISTANT turn, starting from 0>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}
If no emotion is found:
{"turn_index": null, "emotional_word": null, "preceding_context": null, "reasoning": "No emotional language detected"}"""

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
<text>«TEXT»</text>
Respond with ONLY the paraphrased text, nothing else."""


def _extract_json(text: str) -> dict:
    for blob in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            return json.loads(blob.replace("“", '"').replace("”", '"')
                                  .replace("‘", "'").replace("’", "'"))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON in onset output: {text[:200]!r}")


def label_onset(labeller: ChatModel, conversation_text: str) -> dict:
    # Safe substitution (model text may contain literal braces; avoid str.format).
    prompt = ONSET_PROMPT.replace("«CONVERSATION»", conversation_text)
    raw = labeller.chat([{"role": "user", "content": prompt}],
                        temperature=0.0, max_new_tokens=512)
    return _extract_json(raw)


def paraphrase(labeller: ChatModel, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.replace("«TEXT»", text)
    return labeller.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=1024).strip()


def truncate_at_onset(response: str, emotional_word: str | None,
                      preceding_context: str | None) -> str | None:
    """Truncate `response` just before the first emotional word.

    Locates the preceding-context anchor (preferred) or the emotional word, and
    cuts so the prefill ends right before emotion is introduced ("onset"
    truncation). Returns None if neither anchor is found.
    """
    if not emotional_word:
        return None
    if preceding_context and preceding_context in response:
        cut = response.index(preceding_context) + len(preceding_context)
        return response[:cut]
    if emotional_word in response:
        return response[:response.index(emotional_word)]
    return None


def truncate_early(response: str, n_tokens: int = 20) -> str:
    """Whitespace-token "early" truncation (~20 tokens into the turn)."""
    toks = response.split()
    return " ".join(toks[:n_tokens])
