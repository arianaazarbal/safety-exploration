"""Emotion-onset labelling with Claude-Sonnet-4 (verbatim prompt, App. C.1).

Given a conversation, return the assistant-turn index, the short emotional word,
and the immediately-preceding context, used to locate the "onset" truncation
point for the §3 prefill experiment.
"""
from __future__ import annotations

import json
import re

from .. import config_shim as cfg
from ..models.registry import get_judge_client
from ..utils import DiskCache, get_logger, stable_hash

log = get_logger(__name__)

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


def _render_conversation(messages) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


def _last_json(text):
    for blob in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            obj = json.loads(blob)
            if "turn_index" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


class OnsetLabeller:
    def __init__(self, cache_dir=None):
        self.client = get_judge_client()
        self.cache = DiskCache(cache_dir or (cfg.RUNS_DIR / "prefill" / cfg.CACHE_DIRNAME / "onset"))

    def label(self, messages) -> dict | None:
        key = stable_hash({"onset": cfg.ONSET_MODEL, "conv": messages})
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
        raw = self.client.complete(
            model=cfg.ONSET_MODEL, system=None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.0,
        )
        obj = _last_json(raw) or {"turn_index": None}
        self.cache.set(key, obj)
        return obj
