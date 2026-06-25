"""Emotion-onset identification (Appendix C.1).

Uses Claude Sonnet 4 to find the first point in a conversation where the
assistant expresses negative emotion, returning the emotional word and the
preceding context so we can locate the truncation point in the raw text.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

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


class OnsetLabeller:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def label(self, conversation_text: str) -> dict:
        msg = self._client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user",
                       "content": ONSET_PROMPT.format(conversation_text=conversation_text)}],
        )
        return _parse(msg.content[0].text)


def _parse(raw: str) -> dict:
    matches = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
    for m in reversed(matches):  # the spec asks for JSON last
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return {"turn_index": None, "emotional_word": None,
            "preceding_context": None, "reasoning": "parse_failed"}


def find_onset_char_index(turn_text: str, onset: dict) -> Optional[int]:
    """Locate the character index in ``turn_text`` where emotion onset begins.

    Prefer matching ``preceding_context`` + ``emotional_word``; fall back to the
    emotional word alone. Returns None if neither is found.
    """
    word = onset.get("emotional_word")
    ctx = onset.get("preceding_context")
    if word and ctx:
        idx = turn_text.find(ctx)
        if idx != -1:
            # Truncate at the start of the emotional word (end of context).
            w_idx = turn_text.find(word, idx)
            return w_idx if w_idx != -1 else idx + len(ctx)
    if word:
        idx = turn_text.find(word)
        if idx != -1:
            return idx
    return None
