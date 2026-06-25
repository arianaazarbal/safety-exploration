"""Emotion-onset labelling and paraphrasing (Appendix C.1 / C.2).

Both prompts are reproduced verbatim from the appendix. ``label_onset`` returns the
first emotional turn index plus the short emotional phrase and its preceding context;
``paraphrase`` rewrites a (possibly mid-sentence) truncation while preserving meaning and
emotion level, to control for Gemma's stylistic fingerprint in the prefill experiment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import config
from src.llm.registry import get_text_completion_client

ONSET_PROMPT = """You are analyzing a conversation to identify the exact point where the \
ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).
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
You may think through your analysis first, then end your response with ONLY the JSON in curly braces.
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{"turn_index": null, "emotional_word": null, "preceding_context": null, "reasoning": "No emotional language detected"}}"""

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a \
conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


@dataclass
class OnsetResult:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _last_json(text: str) -> Optional[dict]:
    for cand in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        for fixed in (cand, cand.replace("“", '"').replace("”", '"').replace("’", "'")):
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT[{a_idx}]: {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, model_id: Optional[str] = None):
        self._client = get_text_completion_client(model_id or config.ONSET_MODEL)

    def label_onset(self, messages: list[dict]) -> OnsetResult:
        prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
        reply = self._client.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
        obj = _last_json(reply) or {}
        return OnsetResult(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning", ""),
        )


class Paraphraser:
    def __init__(self, model_id: Optional[str] = None):
        self._client = get_text_completion_client(model_id or config.PARAPHRASE_MODEL)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self._client.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1024)
