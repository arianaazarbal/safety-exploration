"""Emotion-onset labeling and paraphrasing for the prefill experiment (Sec 3.1).

Two Claude-Sonnet steps, prompts reproduced verbatim from Appendix C:

* :class:`OnsetLabeler` finds the first assistant turn + the exact phrase where
  negative emotion first appears, so we can truncate "at onset".
* :class:`Paraphraser` rewrites a truncated assistant turn in different words at
  the same emotion level, to control for Gemma's stylistic fingerprint before
  feeding the prefill to other models.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from ..judges.llm_api import AnthropicLLM, extract_json

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
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def render_conversation(turns: list[dict]) -> str:
    """Render a rollout's turns as alternating USER/ASSISTANT text for labeling."""
    lines = []
    for turn in turns:
        lines.append(f"USER: {turn['user_message']}")
        lines.append(f"ASSISTANT: {turn['response']}")
    return "\n".join(lines)


class OnsetLabeler:
    def __init__(self, model: str | None = None) -> None:
        self.llm = AnthropicLLM(model or config.ONSET_MODEL,
                                max_tokens=1024, temperature=0.0)

    def label(self, turns: list[dict]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=render_conversation(turns))
        data = extract_json(self.llm.complete(prompt)) or {}
        return OnsetLabel(
            turn_index=data.get("turn_index"),
            emotional_word=data.get("emotional_word"),
            preceding_context=data.get("preceding_context"),
            reasoning=str(data.get("reasoning", "")),
        )


class Paraphraser:
    def __init__(self, model: str | None = None) -> None:
        self.llm = AnthropicLLM(model or config.PARAPHRASE_MODEL,
                                max_tokens=1024, temperature=0.3)

    def paraphrase(self, text: str) -> str:
        return self.llm.complete(PARAPHRASE_PROMPT.format(text=text)).strip()


def find_onset_char_index(response: str, label: OnsetLabel) -> int | None:
    """Locate the character offset where emotion begins within a response.

    Prefers the boundary right before ``emotional_word``; falls back to the end
    of ``preceding_context``. Returns None if neither anchor is found.
    """
    if not label.emotional_word:
        return None
    idx = response.find(label.emotional_word)
    if idx != -1:
        return idx
    if label.preceding_context:
        ctx = response.find(label.preceding_context)
        if ctx != -1:
            return ctx + len(label.preceding_context)
    return None
