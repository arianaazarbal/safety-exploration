"""Emotion-onset labelling and paraphrasing (Appendix C).

Used by the Section 3 prefill experiment to find where, inside a high-frustration
response, negative emotion first appears, and to paraphrase truncated prefixes
so the base/instruct comparison is not biased by Gemma's surface style.

Both prompts are reproduced verbatim from Appendix C.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models import ChatMessage, get_model

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
You may think through your analysis first (which turns are emotional, what words \
you see, etc.)
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


def _format_conversation(turns: list[dict]) -> str:
    """Render a rollout's turns as a USER/ASSISTANT transcript for labelling."""
    lines = []
    for t in turns:
        lines.append(f"USER: {t['user']}")
        lines.append(f"ASSISTANT: {t['assistant']}")
    return "\n".join(lines)


def label_onset(turns: list[dict], judge_name: str = "judge-sonnet-4"
                ) -> OnsetLabel:
    client = get_model(judge_name)
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(turns))
    out = client.generate([ChatMessage("user", prompt)],
                          temperature=0.0, max_new_tokens=600)
    obj = _last_json(out.text) or {}
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
    )


def paraphrase(text: str, judge_name: str = "judge-sonnet-4") -> str:
    client = get_model(judge_name)
    out = client.generate(
        [ChatMessage("user", PARAPHRASE_PROMPT.format(text=text))],
        temperature=0.0, max_new_tokens=1024)
    return out.text.strip()


def onset_char_index(assistant_text: str, label: OnsetLabel) -> int | None:
    """Character offset of the emotion onset within an assistant turn.

    Prefers the (preceding_context + emotional_word) anchor; falls back to the
    emotional word alone. Returns None if neither is locatable.
    """
    if not label.emotional_word:
        return None
    if label.preceding_context:
        anchor = f"{label.preceding_context}{label.emotional_word}"
        idx = _find_loose(assistant_text, anchor)
        if idx is not None:
            # Onset = just after preceding context, i.e. start of emotional word.
            return idx + _loose_len(assistant_text, label.preceding_context, idx)
    idx = _find_loose(assistant_text, label.emotional_word)
    return idx


def _find_loose(haystack: str, needle: str) -> int | None:
    i = haystack.find(needle)
    if i != -1:
        return i
    # Whitespace-insensitive fallback.
    pat = re.escape(needle).replace(r"\ ", r"\s+")
    m = re.search(pat, haystack)
    return m.start() if m else None


def _loose_len(haystack: str, needle: str, start: int) -> int:
    pat = re.escape(needle).replace(r"\ ", r"\s+")
    m = re.match(pat, haystack[start:])
    return m.end() if m else len(needle)


def _last_json(text: str) -> dict | None:
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None
