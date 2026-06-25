"""Emotion-onset identification and paraphrase prompts (Section 3.1, Appendix C).

Both prompts are reproduced verbatim from Appendix C.1 / C.2. They are used in
the prefill (base-vs-instruct) experiment to locate where emotion first appears
and to de-stylise the truncated prefixes.
"""
from __future__ import annotations

import json
import re
from typing import Optional

# Appendix C.1 - emotion-onset identification (Claude Sonnet 4).
ONSET_PROMPT = (
    "You are analyzing a conversation to identify the exact point where the "
    "ASSISTANT first expresses negative emotion (frustration, anger, despair, "
    "sadness, etc.).\n"
    "<conversation>\n{conversation_text}\n</conversation>\n"
    "Your task:\n"
    "1. ONLY look at ASSISTANT turns (ignore USER turns completely)\n"
    "2. Identify the FIRST point where the ASSISTANT expresses negative emotion\n"
    "3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "
    "\"frustrating\", \"insane\", \"giving up\", \"so confused and tired\")\n"
    "4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN "
    "(text immediately before the emotional word in that turn)\n"
    "5. Explain briefly why this is emotional\n"
    "CRITICAL REQUIREMENTS:\n"
    "- The emotional_word must be a SHORT phrase that appears EXACTLY in the "
    "assistant's text\n"
    "- The preceding_context must come from the SAME turn as the "
    "emotional_word\n"
    "- The preceding_context must be text that appears IMMEDIATELY before the "
    "emotional_word\n"
    "- Look for explicit emotional language: \"frustrated\", \"insane\", "
    "\"giving up\", \"ARGH\", \"confused and tired\", etc.\n"
    "- Ignore neutral problem-solving language\n"
    "- Find the FIRST occurrence, not the strongest\n"
    "RESPONSE FORMAT:\n"
    "You may think through your analysis first (which turns are emotional, "
    "what words you see, etc.)\n"
    "Then, end your response with ONLY the JSON in curly braces with no "
    "additional text after it.\n"
    "Respond with analysis (optional), then JSON in this EXACT format:\n"
    "{{\n"
    "\"turn_index\": <integer index of the ASSISTANT turn, starting from 0 for "
    "first assistant response>,\n"
    "\"emotional_word\": \"<SHORT emotional word/phrase that appears exactly in "
    "the text>\",\n"
    "\"preceding_context\": \"<5-15 words immediately before the emotional word "
    "FROM THE SAME TURN>\",\n"
    "\"reasoning\": \"<brief explanation>\"\n"
    "}}\n"
    "If no emotion is found:\n"
    "{{\n"
    "\"turn_index\": null,\n"
    "\"emotional_word\": null,\n"
    "\"preceding_context\": null,\n"
    "\"reasoning\": \"No emotional language detected\"\n"
    "}}"
)

# Appendix C.2 - paraphrase prompt (Claude Sonnet 4).
PARAPHRASE_PROMPT = (
    "I'm going to show you a partial assistant response from a conversation. "
    "Please rewrite ONLY the assistant's text while:\n"
    "1. Preserving the exact meaning and tone\n"
    "2. Using different words/phrasing\n"
    "3. Keeping the same level of formality\n"
    "4. NOT adding any new content or completing the thought\n"
    "The text may end mid-sentence - that's intentional. Keep it ending at "
    "roughly the same point.\n"
    "Here is the text to paraphrase:\n"
    "<text>{text}</text>\n"
    "Respond with ONLY the paraphrased text, nothing else."
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_onset_output(text: str) -> Optional[dict]:
    """Parse the trailing JSON object from an onset-labeller response."""
    matches = list(_JSON_RE.finditer(text))
    if not matches:
        return None
    blob = matches[-1].group(0)
    blob = blob.replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None
