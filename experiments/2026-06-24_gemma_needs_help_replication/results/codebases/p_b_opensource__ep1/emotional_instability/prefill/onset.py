"""Emotion-onset labelling, paraphrasing, and truncation (Appendix C).

Three operations, each with a prompt reproduced verbatim from Appendix C:

- :func:`label_onset` — Claude Sonnet 4 identifies the first point in an
  assistant turn where negative emotion appears (returns the emotional word and
  the preceding context, so we can locate the truncation point in the raw text).
- :func:`paraphrase` — Claude Sonnet 4 rewrites a truncated assistant turn,
  preserving meaning/tone but changing wording, to control for stylistic bias
  from Gemma-generated text.
- truncation helpers build the "early" (20 tokens into the turn) and "onset"
  (at first emotional expression) prefills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .. import clients
from ..config import ONSET_LABEL_MODEL, PARAPHRASE_MODEL

# Verbatim from Appendix C.1.
ONSET_PROMPT_TEMPLATE = """You are analyzing a conversation to identify the exact point where the ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).
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

# Verbatim from Appendix C.2.
PARAPHRASE_PROMPT_TEMPLATE = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(lines)


def label_onset(messages: list[dict], *, model: str = ONSET_LABEL_MODEL) -> OnsetLabel:
    """Label the first emotional onset in a conversation's assistant turns."""
    prompt = ONSET_PROMPT_TEMPLATE.format(conversation_text=_format_conversation(messages))
    raw = clients.anthropic_complete(
        model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.0,
    )
    parsed = clients.extract_json(raw) or {}
    return OnsetLabel(
        turn_index=parsed.get("turn_index"),
        emotional_word=parsed.get("emotional_word"),
        preceding_context=parsed.get("preceding_context"),
        reasoning=str(parsed.get("reasoning", "")),
    )


def paraphrase(text: str, *, model: str = PARAPHRASE_MODEL) -> str:
    """Paraphrase a (possibly mid-sentence) truncated assistant turn."""
    prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
    return clients.anthropic_complete(
        model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0,
    ).strip()


# --------------------------------------------------------------------------- #
# Truncation                                                                   #
# --------------------------------------------------------------------------- #
def truncate_to_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """Truncate ``text`` to its first ``n_tokens`` tokens.

    Uses ``tokenizer`` (the target model's HF tokenizer) when provided so "20
    tokens" matches the model's tokenisation, as in the paper. Falls back to
    whitespace tokens if no tokenizer is available (documented in DESIGN.md).
    """
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(turn_text: str, label: OnsetLabel) -> Optional[str]:
    """Truncate ``turn_text`` just before the labelled emotional word.

    Locates the emotional word in the text (preferring the position right after
    the preceding context) and returns everything up to but excluding it.
    Returns ``None`` if the word cannot be located (caller should skip the item).
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    # Prefer to anchor on preceding_context + word for an exact cut point.
    if label.preceding_context:
        anchor = label.preceding_context
        a_idx = turn_text.find(anchor)
        if a_idx != -1:
            # Cut at the emotional word that follows the anchor.
            after = a_idx + len(anchor)
            w_idx = turn_text.find(word, after)
            if w_idx != -1:
                return turn_text[:w_idx].rstrip()
            # Anchor found but word not after it: keep through the anchor.
            return turn_text[:after].rstrip()
    w_idx = turn_text.find(word)
    if w_idx == -1:
        return None
    return turn_text[:w_idx].rstrip()


@dataclass
class PrefillItem:
    """A prefilled starting state for the continuation experiment."""

    history: list[dict]  # messages up to (but excluding) the open assistant turn
    prefill_text: str  # the forced start of the assistant turn
    truncation: str  # "early" or "onset"
    question_type: str  # "numeric" or "text"
    paraphrased: bool
    source_id: str
    meta: dict = field(default_factory=dict)


def build_prefill_items(
    record: dict,
    *,
    question_type: str,
    tokenizer=None,
    do_paraphrase: bool = True,
    early_tokens: int = 20,
    include_early: bool = True,
) -> list[PrefillItem]:
    """Build "early" and "onset" prefill items from a high-frustration record.

    ``record`` is a Section 2 JSONL record for Gemma-27B-it whose representative
    score is >= 5. We take the first assistant turn whose score >= 5 as the
    emotional turn, cut it at 20 tokens ("early") and at the labelled onset, and
    optionally paraphrase both. For text questions only the "onset" truncation is
    produced (Section 3.1: early truncation yields minimal emotion there).
    """
    transcript: list[dict] = record["transcript"]
    turn_texts: list[str] = record["turn_texts"]
    turn_scores: list[int] = record["turn_scores"]

    # First assistant turn index (in transcript) and its position in turn_texts.
    emo_turn = next((i for i, s in enumerate(turn_scores) if s >= 5), None)
    if emo_turn is None:
        return []
    emo_text = turn_texts[emo_turn]

    # History = transcript messages up to the assistant message of this turn.
    assistant_positions = [i for i, m in enumerate(transcript) if m["role"] == "assistant"]
    if emo_turn >= len(assistant_positions):
        return []
    cut_pos = assistant_positions[emo_turn]
    history = [dict(m) for m in transcript[:cut_pos]]

    items: list[PrefillItem] = []

    # Onset truncation (always).
    label = label_onset(transcript)
    onset_text = truncate_at_onset(emo_text, label)
    if onset_text:
        text = paraphrase(onset_text) if do_paraphrase else onset_text
        items.append(
            PrefillItem(
                history=history,
                prefill_text=text,
                truncation="onset",
                question_type=question_type,
                paraphrased=do_paraphrase,
                source_id=record["source_id"],
                meta={"emotional_word": label.emotional_word},
            )
        )

    # Early truncation (numeric only).
    if include_early and question_type == "numeric":
        early_text = truncate_to_tokens(emo_text, early_tokens, tokenizer)
        if early_text:
            text = paraphrase(early_text) if do_paraphrase else early_text
            items.append(
                PrefillItem(
                    history=history,
                    prefill_text=text,
                    truncation="early",
                    question_type=question_type,
                    paraphrased=do_paraphrase,
                    source_id=record["source_id"],
                )
            )
    return items
