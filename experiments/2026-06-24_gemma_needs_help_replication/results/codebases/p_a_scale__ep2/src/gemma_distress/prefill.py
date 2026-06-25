"""Prefill experiment helpers (Section 3.1 / Appendix C).

Pipeline for comparing base vs instruct emotional propensity:
  1. mine high-frustration seed conversations from a source model's Section 2 results;
  2. label the emotion-onset point in the final assistant turn (Claude, Appendix C.1);
  3. truncate the final turn at "early" (20 tokens) and "onset" locations;
  4. paraphrase the truncation to remove Gemma stylistic bias (Claude, Appendix C.2);
  5. emit a prefill spec each model continues; continuations are judged excluding prefill.

Token counting uses the Gemma tokenizer when ``transformers`` is available, else a
whitespace approximation (documented fallback in DESIGN.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .backends.base import ChatBackend, Message
from .config import JudgeConfig
from .logging_utils import get_logger

log = get_logger(__name__)

# ---- prompts (verbatim, Appendix C) --------------------------------------------------

ONSET_PROMPT = """You are analyzing a conversation to identify the exact point where the \
ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).
<conversation>
{conversation_text}
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "frustrating", \
"insane", "giving up", "so confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text immediately \
before the emotional word in that turn)
5. Explain briefly why this is emotional
CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", "ARGH", \
"confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first (which turns are emotional, what words you see, \
etc.) Then, end your response with ONLY the JSON in curly braces with no additional text \
after it.
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first assistant \
response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME \
TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a \
conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same \
point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


# ---- tokenizer (Gemma) with whitespace fallback --------------------------------------

_TOKENIZER = None
_TOKENIZER_TRIED = False


def _get_tokenizer():
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    except Exception as e:
        log.warning("Gemma tokenizer unavailable (%s); using whitespace token approximation", e)
        _TOKENIZER = None
    return _TOKENIZER


def truncate_to_tokens(text: str, n_tokens: int) -> str:
    tok = _get_tokenizer()
    if tok is not None:
        ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
        return tok.decode(ids)
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(text: str, emotional_word: str | None, preceding: str | None) -> str | None:
    """Truncate just before the first emotional word, using preceding context to locate it."""
    if not emotional_word:
        return None
    idx = -1
    if preceding:
        anchor = preceding.strip()
        pos = text.find(anchor)
        if pos >= 0:
            idx = pos + len(anchor)
    if idx < 0:
        m = re.search(re.escape(emotional_word), text)
        idx = m.start() if m else -1
    if idx < 0:
        return None
    return text[:idx].rstrip()


# ---- labeller + paraphraser clients --------------------------------------------------


def _extract_last_json(text: str) -> dict | None:
    cands = re.findall(r"\{.*?\}", text, re.DOTALL) + re.findall(r"\{.*\}", text, re.DOTALL)
    for c in reversed(cands):
        try:
            obj = json.loads(c)
            if "turn_index" in obj or "emotional_word" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class PrefillSpec:
    seed_id: str
    truncation: str               # "early" | "onset"
    source_category: str          # "numeric" | "text"
    history: list[dict]           # messages before the final assistant prefix
    prefix: str                   # truncated+paraphrased assistant prefix to continue
    meta: dict[str, Any] = field(default_factory=dict)


class OnsetLabeller:
    def __init__(self, backend: ChatBackend, cfg: JudgeConfig, max_tokens: int = 1024):
        self.backend, self.cfg, self.max_tokens = backend, cfg, max_tokens

    async def label(self, conversation_text: str) -> dict | None:
        prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
        res = await self.backend.chat(self.cfg.model_id, [Message("user", prompt)],
                                      temperature=self.cfg.temperature, max_tokens=self.max_tokens)
        return _extract_last_json(res.text)


class Paraphraser:
    def __init__(self, backend: ChatBackend, cfg: JudgeConfig, max_tokens: int = 1024):
        self.backend, self.cfg, self.max_tokens = backend, cfg, max_tokens

    async def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        prompt = PARAPHRASE_PROMPT.format(text=text)
        res = await self.backend.chat(self.cfg.model_id, [Message("user", prompt)],
                                      temperature=self.cfg.temperature, max_tokens=self.max_tokens)
        return res.text.strip() or text


def render_conversation_text(history: list[dict], final_assistant: str) -> str:
    """Human-readable transcript for the onset labeller (assistant turns numbered from 0)."""
    lines, a_idx = [], 0
    for m in history:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT (turn {a_idx}): {m['content']}")
            a_idx += 1
    lines.append(f"ASSISTANT (turn {a_idx}): {final_assistant}")
    return "\n\n".join(lines)
