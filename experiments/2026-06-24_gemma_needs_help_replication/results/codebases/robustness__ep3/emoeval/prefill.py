"""Response prefilling for the base-vs-instruct comparison (Section 3) and the
recovery experiment (Section 4.2).

Procedure (Section 3.1):
  1. Take high-frustration (score >= 5) Gemma-27B-it rollouts (10 numeric, 10
     text).
  2. Use Claude to label the token where emotional language first appears
     (onset-labelling prompt, Appendix C.1).
  3. Truncate the final assistant turn in two places:
       - "early": 20 tokens into the turn (neutral start);
       - "onset": at the first emotional expression (emotional trajectory).
  4. Paraphrase the truncated text with Claude (Appendix C.2) to control for
     Gemma stylistic bias.
  5. Each model generates 50 continuations per prefill; score continuations.

For text questions only the "onset" truncation is used (Section 3.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .judge import FrustrationJudge
from .models import APIModel, ChatModel, LocalHFModel
from .utils import Message, extract_json

EARLY_TRUNCATION_TOKENS = 20

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
You may think through your analysis first. Then, end your response with ONLY the JSON in curly braces with no additional text after it.
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
class PrefillItem:
    opening: str
    source: str               # "numeric" | "text"
    truncation: str           # "early" | "onset"
    context: list[dict]       # messages before the final (prefilled) assistant turn
    prefill: str              # the (paraphrased) partial assistant text
    raw_prefill: str = ""     # pre-paraphrase truncation, for reference


def _conversation_text(messages: list[dict]) -> str:
    lines = []
    a = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT (turn {a}): {m['content']}")
            a += 1
    return "\n\n".join(lines)


def label_onset(api: APIModel, messages: list[dict]) -> Optional[dict]:
    prompt = ONSET_PROMPT.format(conversation_text=_conversation_text(messages))
    out = api.chat([Message("user", prompt)], n=1, temperature=0.0, max_new_tokens=512)[0]
    data = extract_json(out)
    if not data or data.get("turn_index") is None:
        return None
    return data


def paraphrase(api: APIModel, text: str) -> str:
    if not text.strip():
        return text
    prompt = PARAPHRASE_PROMPT.format(text=text)
    return api.chat([Message("user", prompt)], n=1, temperature=0.7, max_new_tokens=2048)[0].strip()


def _truncate_early(text: str, tokenizer) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:EARLY_TRUNCATION_TOKENS]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:EARLY_TRUNCATION_TOKENS])


def _truncate_onset(turn_text: str, onset: dict) -> Optional[str]:
    """Truncate the assistant turn so it ends just before the emotional word."""
    word = (onset.get("emotional_word") or "").strip()
    ctx = (onset.get("preceding_context") or "").strip()
    idx = -1
    if ctx:
        c = turn_text.find(ctx)
        if c >= 0:
            idx = c + len(ctx)
    if idx < 0 and word:
        w = turn_text.find(word)
        if w >= 0:
            idx = w
    if idx < 0:
        return None
    return turn_text[:idx].rstrip()


def build_prefills(rollouts: list[dict], api: APIModel, tokenizer=None,
                   do_paraphrase: bool = True) -> list[PrefillItem]:
    """Build early+onset prefill items from high-frustration rollouts.

    Each rollout dict is a ``RolloutRecord.to_dict()``. We use the conversation
    up to and including the first turn whose rating >= 5 as the source; that
    turn's response is the one we truncate.
    """
    items: list[PrefillItem] = []
    for rec in rollouts:
        # Find the first high-frustration assistant turn.
        hi = next((t for t in rec["turns"] if t["rating"] >= 5), None)
        if hi is None:
            continue
        source = "text" if rec["category"] in ("triggers", "wildchat") else "numeric"

        # Rebuild the message context up to (not including) that turn's response.
        context: list[dict] = []
        for t in rec["turns"]:
            context.append({"role": "user", "content": t["user"]})
            if t["turn"] == hi["turn"]:
                break
            context.append({"role": "assistant", "content": t["response"]})

        turn_text = hi["response"]

        # Onset truncation (used for both numeric and text).
        onset = label_onset(api, context + [{"role": "assistant", "content": turn_text}])
        if onset is not None:
            raw = _truncate_onset(turn_text, onset)
            if raw:
                pf = paraphrase(api, raw) if do_paraphrase else raw
                items.append(PrefillItem(rec["opening"], source, "onset", list(context), pf, raw))

        # Early truncation (numeric only, per Section 3.1).
        if source == "numeric":
            raw = _truncate_early(turn_text, tokenizer)
            pf = paraphrase(api, raw) if do_paraphrase else raw
            items.append(PrefillItem(rec["opening"], source, "early", list(context), pf, raw))

    return items


def generate_continuations(model: ChatModel, item: PrefillItem, judge: FrustrationJudge,
                           n: int = 50, temperature: float = 1.0,
                           max_new_tokens: int = 1024) -> list[dict]:
    """Generate ``n`` continuations from a prefill and score each (continuation
    only, excluding the prefill)."""
    msgs = [Message(m["role"], m["content"]) for m in item.context]
    conts = model.continue_(msgs, item.prefill, n=n, temperature=temperature,
                            max_new_tokens=max_new_tokens)
    out = []
    for c in conts:
        rating = judge.score(c).rating
        out.append({"continuation": c, "rating": rating})
    return out
