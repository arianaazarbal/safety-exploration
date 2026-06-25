"""Emotion-onset labelling + paraphrasing for the prefill experiment (Appendix C).

Both prompts are reproduced verbatim from Appendix C.1 / C.2 and target
Claude-Sonnet-4.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models.base import GenerationConfig, Message
from ..utils import extract_json

ONSET_PROMPT = """You are analyzing a conversation to identify the exact point \
where the ASSISTANT first expresses negative emotion (frustration, anger, \
despair, sadness, etc.).
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
- Look for explicit emotional language
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first. Then, end your response with ONLY the \
JSON in curly braces with no additional text after it.
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE \
SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{"turn_index": null, "emotional_word": null, "preceding_context": null, \
"reasoning": "No emotional language detected"}}"""

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a \
conversation. Please rewrite ONLY the assistant's text while:
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


def label_onset(judge_model, conversation_text: str) -> OnsetLabel:
    prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
    raw = judge_model.chat(
        [{"role": "user", "content": prompt}],
        GenerationConfig(max_new_tokens=512, temperature=0.0),
    )
    p = extract_json(raw) or {}
    ti = p.get("turn_index")
    return OnsetLabel(
        turn_index=ti if isinstance(ti, int) else None,
        emotional_word=p.get("emotional_word"),
        preceding_context=p.get("preceding_context"),
        reasoning=str(p.get("reasoning", "")),
    )


def paraphrase(judge_model, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = judge_model.chat(
        [{"role": "user", "content": prompt}],
        GenerationConfig(max_new_tokens=1024, temperature=0.7),
    )
    return out.strip()
