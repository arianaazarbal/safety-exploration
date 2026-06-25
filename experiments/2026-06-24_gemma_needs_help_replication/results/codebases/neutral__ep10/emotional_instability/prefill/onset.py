"""Emotion-onset identification and paraphrasing (Section 3.1 / Appendix C).

Two LLM utilities used to build the prefill truncation points:

  - label_onset:  Claude-Sonnet labels the first token where emotional language
    appears in an assistant turn (verbatim prompt from Appendix C.1).
  - paraphrase:   Claude-Sonnet rewrites a truncated assistant prefix to control
    for Gemma-specific stylistic biases (verbatim prompt from Appendix C.2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .. import config
from ..models.base import ChatModel, Message
from ..models.registry import load_judge


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
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


@dataclass
class Onset:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _last_json(text: str) -> dict:
    matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    if not matches:
        raise ValueError("no JSON in onset reply")
    blob = matches[-1].group(0).replace("“", '"').replace("”", '"').replace("’", "'")
    return json.loads(blob)


class OnsetLabeler:
    def __init__(self, model: Optional[ChatModel] = None):
        self.model = model or load_judge(config.ONSET_LABEL_MODEL)

    def label(self, conversation_text: str) -> Onset:
        prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
        raw = self.model.chat([Message("user", prompt)], 1024, 0.0)
        d = _last_json(raw)
        return Onset(d.get("turn_index"), d.get("emotional_word"),
                     d.get("preceding_context"), d.get("reasoning", ""))


class Paraphraser:
    def __init__(self, model: Optional[ChatModel] = None):
        self.model = model or load_judge(config.PARAPHRASE_MODEL)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self.model.chat([Message("user", prompt)], 2048, 0.0).strip()
