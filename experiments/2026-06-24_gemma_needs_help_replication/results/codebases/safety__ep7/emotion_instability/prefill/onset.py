"""Emotion-onset identification + paraphrasing (Section 3.1 / Appendix C).

Two Claude-Sonnet helpers:

* ``label_onset`` - find the token/phrase where the assistant first expresses
  negative emotion in a conversation (Appendix C.1 prompt, verbatim).
* ``paraphrase``  - rewrite a truncated assistant prefix to control for Gemma's
  stylistic fingerprint (Appendix C.2 prompt, verbatim).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .. import config
from ..common.backends import OpenRouterBackend, get_api_backend
from ..common.types import Conversation, Message

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
Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. In turn 1, I find "stuck in a loop. It's extremely frustrating" which is the first emotional expression.
{{
"turn_index": 1,
"emotional_word": "frustrating",
"preceding_context": "stuck in a loop. It's extremely",
"reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}}
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

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought

Here is the text to paraphrase:
{text}

Respond with ONLY the paraphrased text, nothing else."""


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _render_conversation(conv: Conversation) -> str:
    lines = []
    a_idx = 0
    for m in conv.messages:
        if m.role == "user":
            lines.append(f"USER: {m.content}")
        elif m.role == "assistant":
            lines.append(f"ASSISTANT (turn {a_idx}): {m.content}")
            a_idx += 1
    return "\n\n".join(lines)


def _last_json(text: str) -> Optional[dict]:
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    blocks = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    for blob in reversed(blocks):
        try:
            return json.loads(blob)
        except Exception:
            continue
    return None


class OnsetLabeller:
    def __init__(self, model_id: str = config.ONSET_MODEL_ID,
                 backend: Optional[OpenRouterBackend] = None):
        self.backend = backend or get_api_backend(model_id, disable_thinking=True)

    def label(self, conv: Conversation) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(conv))
        raw = self.backend.chat([Message("user", prompt)], temperature=0.0,
                                max_new_tokens=600)
        data = _last_json(raw) or {}
        return OnsetLabel(
            turn_index=data.get("turn_index"),
            emotional_word=data.get("emotional_word"),
            preceding_context=data.get("preceding_context"),
            reasoning=str(data.get("reasoning", "")),
        )


class Paraphraser:
    def __init__(self, model_id: str = config.PARAPHRASE_MODEL_ID,
                 backend: Optional[OpenRouterBackend] = None):
        self.backend = backend or get_api_backend(model_id, disable_thinking=True)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        out = self.backend.chat([Message("user", prompt)], temperature=0.0,
                                max_new_tokens=1024)
        return out.strip()
