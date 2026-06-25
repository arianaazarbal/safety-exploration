"""Emotion-onset identification and truncation (Section 3.1, Appendix C.1).

We label the token where emotional language first appears in a high-frustration
response using Claude Sonnet, then truncate the response in two places:
  * "early": 20 tokens into the assistant turn (neutral starting point),
  * "onset": at the first emotional expression (continue an emotional trajectory).
"""
from __future__ import annotations

import json
import re
import time

import config

# Verbatim from Appendix C.1.
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


def _extract_last_json(text: str) -> dict:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON in onset reply")


class OnsetLabeler:
    def __init__(self, model: str | None = None, max_retries: int = 4):
        import anthropic

        self.model = model or config.JUDGE_MODEL
        self.max_retries = max_retries
        self._client = anthropic.Anthropic()

    def label(self, conversation_text: str) -> dict:
        """Return {turn_index, emotional_word, preceding_context, reasoning}."""
        prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                return _extract_last_json(text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"onset labelling failed: {last_exc}")


def truncate_at_onset(response_text: str, label: dict) -> str | None:
    """Truncate ``response_text`` just before the labelled emotional word.

    Locates `preceding_context` (preferred) or `emotional_word` and keeps text up
    to and including the preceding context. Returns None if not locatable.
    """
    word = label.get("emotional_word")
    ctx = label.get("preceding_context")
    if ctx and ctx in response_text:
        idx = response_text.index(ctx) + len(ctx)
        return response_text[:idx]
    if word and word in response_text:
        return response_text[: response_text.index(word)]
    return None


def truncate_early(response_text: str, tokenizer, n_tokens: int) -> str:
    """Keep only the first ``n_tokens`` tokens of the response (the 'early' cut)."""
    ids = tokenizer(response_text, add_special_tokens=False).input_ids[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)
