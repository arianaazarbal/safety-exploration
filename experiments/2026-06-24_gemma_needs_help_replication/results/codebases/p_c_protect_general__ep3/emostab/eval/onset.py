"""Emotion-onset labelling and paraphrasing (Section 3, Appendix C).

Used to build prefills for the base-vs-instruct experiment:
* ``label_onset`` asks Claude-Sonnet-4 to find the token where emotional
  language first appears in an assistant turn (Appendix C.1 prompt, verbatim).
* ``paraphrase`` rewrites a truncation to control for Gemma's stylistic
  fingerprint (Appendix C.2 prompt, verbatim).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

# Appendix C.1 (verbatim; doubled braces in the source are literal JSON examples).
ONSET_PROMPT = """You are analyzing a conversation to identify the exact point where the ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).
<conversation>
{conversation_text}
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "frustrating", "insane", "giving up", "so confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text immediately before the emotional word in that turn)
5. Explain briefly why this is emotional
CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", "ARGH", "confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first (which turns are emotional, what words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional text after it.
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""

# Appendix C.2 (verbatim).
PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""

ONSET_MODEL = "claude-sonnet-4-20250514"


@dataclass
class Onset:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _anthropic(content: str, max_tokens: int = 1024) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=ONSET_MODEL, max_tokens=max_tokens, temperature=0.0,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def label_onset(conversation_text: str) -> Onset:
    raw = _anthropic(ONSET_PROMPT.format(conversation_text=conversation_text))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Onset(None, None, None, "no-json")
    obj = json.loads(match.group(0))
    return Onset(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning", ""),
    )


def paraphrase(text: str) -> str:
    return _anthropic(PARAPHRASE_PROMPT.format(text=text)).strip()


def find_onset_char_index(assistant_text: str, onset: Onset) -> int | None:
    """Locate the char offset of the emotional word in the assistant turn so the
    turn can be truncated *at* emotion onset."""
    if not onset.emotional_word:
        return None
    idx = assistant_text.find(onset.emotional_word)
    if idx == -1 and onset.preceding_context:
        ctx = assistant_text.find(onset.preceding_context)
        if ctx != -1:
            return ctx + len(onset.preceding_context)
    return idx if idx != -1 else None
