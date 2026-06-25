"""Emotion-onset labelling (Section 3.1 / Appendix C.1).

Claude Sonnet identifies the first point in the assistant's turns where negative
emotion appears, returning the turn index, a short emotional word/phrase, and the
preceding context. The prompt is reproduced verbatim from Appendix C.1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..models.anthropic_backend import AnthropicClient

ONSET_PROMPT = """\
You are analyzing a conversation to identify the exact point where the ASSISTANT \
first expresses negative emotion (frustration, anger, despair, sadness, etc.).
<conversation>
{conversation_text}
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "frustrating", \
"insane", "giving up", "so confused and tired")
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
You may think through your analysis first (which turns are emotional, what words \
you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional \
text after it.
Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. In \
turn 1, I find "stuck in a loop. It's extremely frustrating" which is the first \
emotional expression.
{{
"turn_index": 1,
"emotional_word": "frustrating",
"preceding_context": "stuck in a loop. It's extremely",
"reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}}
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE \
SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""


@dataclass(frozen=True)
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: Optional[str]

    @property
    def found(self) -> bool:
        return self.turn_index is not None and bool(self.emotional_word)


_LAST_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)


class OnsetLabeller:
    def __init__(self, model_id: str, max_tokens: int = 1024):
        self._client = AnthropicClient(model_id)
        self._max_tokens = max_tokens

    def label(self, conversation_text: str) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
        raw = self._client.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=self._max_tokens,
        )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> OnsetLabel:
        # The prompt asks for trailing JSON; take the last {...} block.
        match = _LAST_JSON_RE.search(raw.strip())
        if not match:
            # Fall back to the first balanced-looking object.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return OnsetLabel(None, None, None, "unparseable onset output")
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return OnsetLabel(None, None, None, "invalid JSON in onset output")
        return OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning"),
        )
