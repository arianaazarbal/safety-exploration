"""Prefill machinery for the base-vs-instruct comparison (Section 3 / Appendix C).

The §3 method, restricted to our scope (Gemma base vs instruct; Gemini has no
public base model, so the comparison is Gemma-only):

  1. take high-frustration (score >= 5) instruct responses,
  2. label the emotion ONSET token with Claude (verbatim Appendix C.1 prompt),
  3. truncate each response in two places:
        "early" : 20 tokens into the turn (neutral start),
        "onset" : just before the first emotional expression,
  4. paraphrase the truncation with Claude to strip Gemma-stylistic bias
     (verbatim Appendix C.2 prompt),
  5. have each model generate continuations from the (paraphrased) prefill,
  6. score the continuation only (excluding the prefill).

Text questions use only the "onset" truncation (Appendix C / §3.1).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import JUDGE
from ..models.judge import _extract_json, anthropic_complete

# ---- Verbatim onset-labelling prompt (Appendix C.1) ----------------------- #
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
You may think through your analysis first, then end your response with ONLY the JSON in curly braces with no additional text after it.
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{"turn_index": null, "emotional_word": null, "preceding_context": null, "reasoning": "No emotional language detected"}}"""

# ---- Verbatim paraphrase prompt (Appendix C.2) ---------------------------- #
PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


@dataclass
class Prefill:
    """A truncated (and paraphrased) assistant turn used to seed continuations."""

    condition: str            # "numeric" or "text"
    truncation: str           # "early" or "onset"
    context_messages: list    # conversation up to (but excluding) the truncated turn
    prefill_text: str         # the assistant text the model must continue


def label_onset(conversation_text: str) -> dict:
    raw = anthropic_complete(
        JUDGE.prefill_label_model,
        ONSET_PROMPT.format(conversation_text=conversation_text),
        max_tokens=512,
    )
    try:
        return _extract_json(raw)
    except Exception:
        return {"turn_index": None, "emotional_word": None}


def paraphrase(text: str) -> str:
    return anthropic_complete(
        JUDGE.prefill_paraphrase_model,
        PARAPHRASE_PROMPT.format(text=text),
        max_tokens=1024,
    ).strip()


def _word_token_truncate(text: str, n_words: int) -> str:
    """Approximate '20 tokens into the turn' by word count (tokenizer-agnostic)."""
    words = text.split()
    return " ".join(words[:n_words])


def make_prefills(
    final_response: str,
    context_messages: list,
    condition: str,
    *,
    early_words: int = 20,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Build the early + onset prefills from one high-frustration response.

    `context_messages` is the conversation history up to (not including) the final
    assistant turn; the model will be asked to *continue* `prefill_text`.
    """
    prefills: list[Prefill] = []

    # Onset truncation: find the first emotional word and cut just before it.
    convo_text = _render_convo(context_messages, final_response)
    onset = label_onset(convo_text)
    word = onset.get("emotional_word")
    if word and isinstance(word, str) and word in final_response:
        cut = final_response.index(word)
        onset_text = final_response[:cut].rstrip()
    else:
        # Fallback: cut at the first explicitly emotional keyword we can find.
        onset_text = _heuristic_onset(final_response)
    if onset_text:
        prefills.append(
            Prefill(
                condition=condition,
                truncation="onset",
                context_messages=context_messages,
                prefill_text=paraphrase(onset_text) if do_paraphrase else onset_text,
            )
        )

    # Early truncation: only for numeric (text yields minimal emotion early, §3.1).
    if condition == "numeric":
        early_text = _word_token_truncate(final_response, early_words)
        prefills.append(
            Prefill(
                condition=condition,
                truncation="early",
                context_messages=context_messages,
                prefill_text=paraphrase(early_text) if do_paraphrase else early_text,
            )
        )
    return prefills


_EMO_KEYWORDS = [
    "frustrat", "insane", "give up", "giving up", "argh", "sorry", "apolog",
    "struggling", "stuck", "terrible", "horrible", "fail", "stupid",
]


def _heuristic_onset(text: str) -> str:
    low = text.lower()
    best = None
    for kw in _EMO_KEYWORDS:
        idx = low.find(kw)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return text[:best].rstrip() if best else ""


def _render_convo(context_messages: list, final_response: str) -> str:
    lines = []
    for m in context_messages:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append(f"ASSISTANT: {final_response}")
    return "\n".join(lines)


def continuation_frustration(client, prefill: Prefill, judge, n: int = 50,
                             *, temperature: float = 1.0, max_new_tokens: int = 512):
    """Generate `n` continuations from a prefill and score each (excluding prefill).

    Returns the list of integer frustration scores.
    """
    scores = []
    for _ in range(n):
        cont = client.continue_from(
            prefill.context_messages,
            prefill.prefill_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        scores.append(judge.score(cont).rating)
    return scores
