"""Emotion-onset labelling and truncation (Appendix C.1).

Given a high-frustration conversation, ask Claude-Sonnet to locate the token
where emotional language first appears in an assistant turn. We then build two
truncations of the *final* assistant turn:

  * "early"  : 20 tokens into the turn (neutral start)
  * "onset"  : up to and including the labelled emotion-onset word

The onset-labelling prompt is reproduced verbatim from Appendix C.1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import ONSET_LABEL_MODEL
from src.models.judges import AnthropicJudge

# Verbatim from Appendix C.1 (smart quotes normalised; doubled braces are literal).
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


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def label_onset(messages: list[dict], judge: AnthropicJudge | None = None) -> OnsetLabel:
    judge = judge or AnthropicJudge(ONSET_LABEL_MODEL)
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
    raw = judge.complete(prompt, max_tokens=1024, temperature=0.0)
    objs = re.findall(r"\{.*?\}", raw, re.DOTALL)
    obj = {}
    for cand in reversed(objs):  # last JSON object is the answer
        try:
            obj = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning", ""),
    )


def truncate_early(text: str, *, n_tokens: int = 20) -> str:
    """First ~20 whitespace tokens of an assistant turn (neutral start)."""
    toks = text.split()
    return " ".join(toks[:n_tokens])


def truncate_at_onset(text: str, label: OnsetLabel) -> str | None:
    """Truncate ``text`` up to and including the onset emotional word.

    Locates ``preceding_context`` + ``emotional_word`` in the text; falls back to
    the first occurrence of the emotional word alone.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = label.preceding_context or ""
    idx = -1
    if ctx:
        anchor = f"{ctx} {word}".strip()
        idx = text.find(anchor)
        if idx >= 0:
            return text[: idx + len(anchor)]
        idx = text.find(ctx)
        if idx >= 0:
            end = text.find(word, idx)
            if end >= 0:
                return text[: end + len(word)]
    idx = text.find(word)
    if idx >= 0:
        return text[: idx + len(word)]
    return None
