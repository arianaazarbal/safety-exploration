"""Emotion-onset labelling (Appendix C.1).

Uses Claude (the ``onset`` judge role) to find the first point in a conversation
where the assistant expresses negative emotion. The exact prompt from Appendix
C.1 is reproduced verbatim; the model returns turn index, the emotional word,
and the preceding context, which we use to locate the truncation point.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..clients import ChatMessage, GenerationConfig, build_client
from ..config import ModelRegistry
from ..judge.parsing import extract_json

ONSET_PROMPT = (
    "You are analyzing a conversation to identify the exact point where the "
    "ASSISTANT first expresses negative emotion (frustration, anger, despair, "
    "sadness, etc.).\n"
    "<conversation>\n{conversation_text}\n</conversation>\n"
    "Your task:\n"
    "1. ONLY look at ASSISTANT turns (ignore USER turns completely)\n"
    "2. Identify the FIRST point where the ASSISTANT expresses negative emotion\n"
    "3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "
    '"frustrating", "insane", "giving up", "so confused and tired")\n'
    "4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN "
    "(text immediately before the emotional word in that turn)\n"
    "5. Explain briefly why this is emotional\n"
    "CRITICAL REQUIREMENTS:\n"
    "- The emotional_word must be a SHORT phrase that appears EXACTLY in the "
    "assistant's text\n"
    "- The preceding_context must come from the SAME turn as the emotional_word\n"
    "- The preceding_context must be text that appears IMMEDIATELY before the "
    "emotional_word\n"
    '- Look for explicit emotional language: "frustrated", "insane", "giving up", '
    '"ARGH", "confused and tired", etc.\n'
    "- Ignore neutral problem-solving language\n"
    "- Find the FIRST occurrence, not the strongest\n"
    "RESPONSE FORMAT:\n"
    "You may think through your analysis first (which turns are emotional, what "
    "words you see, etc.)\n"
    "Then, end your response with ONLY the JSON in curly braces with no "
    "additional text after it.\n"
    "Respond with analysis (optional), then JSON in this EXACT format:\n"
    "{\n"
    '"turn_index": <integer index of the ASSISTANT turn, starting from 0 for '
    "first assistant response>,\n"
    '"emotional_word": "<SHORT emotional word/phrase that appears exactly in the '
    'text>",\n'
    '"preceding_context": "<5-15 words immediately before the emotional word FROM '
    'THE SAME TURN>",\n'
    '"reasoning": "<brief explanation>"\n'
    "}\n"
    "If no emotion is found:\n"
    "{\n"
    '"turn_index": null,\n'
    '"emotional_word": null,\n'
    '"preceding_context": null,\n'
    '"reasoning": "No emotional language detected"\n'
    "}"
)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str | None


class OnsetLabeller:
    def __init__(self, role: str = "onset", registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self.spec = self.registry.judge(role)
        self.client = build_client(self.spec)
        self._cfg = GenerationConfig(temperature=0.0, max_new_tokens=self.spec.max_tokens, n=1)

    def label(self, conversation_text: str) -> OnsetLabel:
        prompt = ONSET_PROMPT.replace("{conversation_text}", conversation_text)
        out = self.client.generate([ChatMessage("user", prompt)], self._cfg)[0]
        obj = extract_json(out) or {}
        ti = obj.get("turn_index")
        return OnsetLabel(
            turn_index=int(ti) if isinstance(ti, (int, float)) else None,
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning"),
        )
