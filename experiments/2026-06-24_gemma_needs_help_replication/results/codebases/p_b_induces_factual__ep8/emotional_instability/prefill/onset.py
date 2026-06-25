"""Emotion-onset identification (Appendix C.1).

Claude-Sonnet labels the token where emotional language first appears in a
high-frustration conversation, so each response can be truncated at "onset". The
prompt is transcribed verbatim from Appendix C.1.
"""

from __future__ import annotations

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

import config

from ..models.base import ChatMessage
from ..utils import extract_json

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
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
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
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _render_conversation(messages: list[ChatMessage]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, model: str = config.ONSET_MODEL):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def label(self, messages: list[ChatMessage]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
        msg = self._client.messages.create(
            model=self.model, max_tokens=1024, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        obj = extract_json(text)
        return OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=str(obj.get("reasoning", "")),
        )
