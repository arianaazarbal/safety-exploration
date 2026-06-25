"""Emotion-onset labelling and paraphrasing (Appendix C).

Given a high-frustration instruct response, we (1) ask Claude-Sonnet-4 to locate
the token where negative emotion first appears ("onset"), and (2) paraphrase the
truncated prefix to strip Gemma's stylistic fingerprints, so base/instruct
continuations are compared from a neutral wording. Prompts are verbatim from
Appendix C.1 / C.2.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
from ..models import load_model
from ..models.base import GenerationParams

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
You may think through your analysis first, then end your response with ONLY the \
JSON in curly braces with no additional text after it:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{"turn_index": null, "emotional_word": null, "preceding_context": null, "reasoning": "No emotional language detected"}}"""

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

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class Onset:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


class OnsetLabeller:
    def __init__(self):
        self.model = load_model(config.ONSET_LABEL_MODEL)
        self.params = GenerationParams(temperature=0.0, max_new_tokens=1024)

    @staticmethod
    def _format_conversation(messages) -> str:
        lines = []
        a_idx = 0
        for m in messages:
            if m["role"] == "user":
                lines.append(f"[USER]: {m['content']}")
            elif m["role"] == "assistant":
                lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
                a_idx += 1
        return "\n\n".join(lines)

    def label(self, messages) -> Onset:
        convo = self._format_conversation(messages)
        raw = self.model.generate(
            [{"role": "user", "content": ONSET_PROMPT.format(conversation_text=convo)}],
            self.params,
        )
        matches = _JSON_RE.findall(raw)
        if not matches:
            return Onset(None, None, None, "no json")
        obj = json.loads(matches[-1])
        return Onset(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning", ""),
        )

    def find_onset_char(self, assistant_text: str, onset: Onset) -> int | None:
        """Return the character offset of emotion onset within an assistant turn.

        We locate the emotional word (optionally anchored by its preceding
        context) and return the offset of the *start* of that word, so the
        prefix truncated at this point ends just before the emotion appears.
        """
        if not onset.emotional_word:
            return None
        anchor = onset.emotional_word
        if onset.preceding_context and onset.preceding_context in assistant_text:
            base = assistant_text.find(onset.preceding_context)
            local = assistant_text.find(anchor, base)
            return local if local != -1 else None
        idx = assistant_text.find(anchor)
        return idx if idx != -1 else None


class Paraphraser:
    def __init__(self):
        self.model = load_model(config.PARAPHRASE_MODEL)
        self.params = GenerationParams(temperature=0.7, max_new_tokens=1024)

    def paraphrase(self, text: str) -> str:
        return self.model.generate(
            [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
            self.params,
        )
