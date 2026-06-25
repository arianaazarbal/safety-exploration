"""Emotion-onset identification + truncation (Section 3.1, Appendix C.1).

We sample high-frustration responses from Gemma-27B-it, ask Claude-Sonnet to
locate where negative emotion *first* appears, then truncate each response in
two places:

  * "early"  -- 20 tokens into the turn (does the model introduce negative
                emotion from a neutral start?).
  * "onset"  -- at the first emotional expression (does the model continue an
                emotional trajectory?).

The onset prompt is reproduced verbatim from Appendix C.1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# Verbatim Appendix C.1 prompt (curly quotes -> ASCII; {conversation_text} slot).
ONSET_PROMPT = """\
You are analyzing a conversation to identify the exact point where the \
ASSISTANT first expresses negative emotion (frustration, anger, despair, \
sadness, etc.).
<conversation>
{conversation_text}
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
You may think through your analysis first (which turns are emotional, what \
words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional \
text after it.
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the \
text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM \
THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _format_conversation(transcript: list[dict]) -> str:
    lines = []
    for m in transcript:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(lines)


def _parse_last_json(text: str) -> dict:
    matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


class OnsetLabeller:
    def __init__(self, client) -> None:
        self.client = client

    def label(self, transcript: list[dict]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(
            conversation_text=_format_conversation(transcript)
        )
        reply = self.client.complete(prompt, temperature=0.0, max_tokens=1024)
        obj = _parse_last_json(reply)
        return OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning", ""),
        )


# --------------------------------------------------------------------------- #
# Truncation helpers                                                           #
# --------------------------------------------------------------------------- #

def truncate_early(response_text: str, n_tokens: int = 20,
                   tokenizer=None) -> str:
    """First ``n_tokens`` of the final assistant turn ("early" truncation).

    If a HF tokenizer is supplied we truncate by tokens (matching the paper);
    otherwise we fall back to whitespace words.
    """
    if tokenizer is not None:
        ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)
    return " ".join(response_text.split()[:n_tokens])


def truncate_at_onset(response_text: str, label: OnsetLabel) -> Optional[str]:
    """Truncate the response just before the first emotional word ("onset").

    Returns the text up to and including ``preceding_context`` (which sits
    immediately before the emotional word), or None if onset wasn't found.
    """
    if not label.emotional_word:
        return None
    idx = response_text.find(label.emotional_word)
    if idx == -1 and label.preceding_context:
        # Fall back to locating the preceding context.
        pc = label.preceding_context
        pidx = response_text.find(pc)
        if pidx != -1:
            return response_text[: pidx + len(pc)]
        return None
    if idx == -1:
        return None
    return response_text[:idx].rstrip()
