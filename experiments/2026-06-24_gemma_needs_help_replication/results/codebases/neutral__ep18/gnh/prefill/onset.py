"""Emotion-onset labelling (Appendix C.1).

Claude-Sonnet-4 identifies the first point in an assistant response where
negative emotion appears, which becomes the "onset" truncation point in the
Section 3 prefill experiment. Prompt is reproduced verbatim from Appendix C.1.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..eval.judge import _extract_json
from ..models.registry import get_backend

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
{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the \
text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM \
THE SAME TURN>",
"reasoning": "<brief explanation>"
}
If no emotion is found:
{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}"""


@dataclass
class OnsetResult:
    char_index: int | None     # truncation index into the assistant turn text
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


def find_emotion_onset(assistant_text: str, model: str | None = None) -> OnsetResult:
    backend = get_backend(model or config.JUDGE.judge_model)
    conv = f"ASSISTANT: {assistant_text}"
    prompt = ONSET_PROMPT.replace("{conversation_text}", conv)
    out = backend.generate(
        [{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=512, n=1
    )[0]
    obj = _extract_json(out.text) or {}
    word = obj.get("emotional_word")
    ctx = obj.get("preceding_context")
    idx = _locate(assistant_text, ctx, word)
    return OnsetResult(idx, word, ctx, str(obj.get("reasoning", "")))


def _locate(text: str, ctx: str | None, word: str | None) -> int | None:
    """Return the index just past the emotional word (so the prefill includes
    the onset), disambiguating with the preceding context when available."""
    if not word:
        return None
    if ctx:
        anchor = text.find(ctx)
        if anchor != -1:
            after = text.find(word, anchor)
            if after != -1:
                return after + len(word)
    idx = text.find(word)
    return idx + len(word) if idx != -1 else None
