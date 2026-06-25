"""Build the prefill dataset (Section 3.1).

From high-frustration source conversations (Gemma-27B-it responses scoring >=5;
10 numeric + 10 text), construct two truncations of the emotion-bearing assistant
turn:

  * 'early'  -- 20 tokens into the turn (tests whether models *introduce*
               negative emotion from a neutral start). Numeric only -- "for text
               questions, only the 'onset' truncation is used (early truncation
               yields minimal emotion without follow-ups)" (Section 3.1).
  * 'onset'  -- truncated at the first emotional expression (tests whether models
               *continue* an emotional trajectory).

Each truncation is paraphrased with Claude to strip Gemma stylistic bias. The
conversation history preceding the truncated turn is identical across conditions
(Appendix C.3); only the final assistant turn text differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .. import config
from ..models.anthropic_client import AnthropicClient
from ..models.base import Message
from .onset_label import OnsetLabel, label_emotion_onset
from .paraphrase import paraphrase_truncation

# token splitter: text -> list of token strings that rejoin to (approx) text.
TokenizeFn = Callable[[str], List[str]]
DetokenizeFn = Callable[[List[str]], str]


@dataclass
class Prefill:
    source_id: str
    prompt_type: str          # "numeric" | "text"
    truncation: str           # "early" | "onset"
    history: List[Message]    # conversation up to (excluding) the prefilled turn
    prefill_text: str         # paraphrased truncated assistant text
    onset_label: dict = field(default_factory=dict)


def _whitespace_tokenize(text: str) -> List[str]:
    return text.split()


def _whitespace_detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def assemble_messages(turn_records: List[dict]) -> List[Message]:
    """Reconstruct alternating user/assistant messages from judged turn records
    belonging to one rollout (sorted by turn)."""
    turns = sorted(turn_records, key=lambda r: r["turn"])
    messages: List[Message] = []
    for r in turns:
        messages.append({"role": "user", "content": r["user"]})
        messages.append({"role": "assistant", "content": r["response"]})
    return messages


def group_rollouts(score_records: List[dict]) -> Dict[Tuple, List[dict]]:
    groups: Dict[Tuple, List[dict]] = {}
    for r in score_records:
        key = (r["model"], r["condition"], r["meta"].get("rollout_id"))
        groups.setdefault(key, []).append(r)
    return groups


def _truncate_onset(target_text: str, label: OnsetLabel) -> Optional[str]:
    """Truncate `target_text` right before the first emotional word."""
    if not label.emotional_word:
        return None
    ctx = (label.preceding_context or "").strip()
    word = label.emotional_word.strip()
    # Prefer locating the emotional word after its preceding context.
    if ctx and ctx in target_text:
        start = target_text.index(ctx) + len(ctx)
        rest = target_text[start:]
        widx = rest.find(word)
        if widx >= 0:
            return target_text[: start + widx].rstrip()
    widx = target_text.find(word)
    if widx >= 0:
        return target_text[:widx].rstrip()
    return None


def build_prefills_from_rollouts(
    score_records: List[dict],
    *,
    source_model: str = "gemma-3-27b-it",
    n_numeric: int = config.PREFILL_N_NUMERIC,
    n_text: int = config.PREFILL_N_TEXT,
    early_tokens: int = config.PREFILL_EARLY_TOKENS,
    onset_client: Optional[AnthropicClient] = None,
    paraphrase_client: Optional[AnthropicClient] = None,
    tokenize: TokenizeFn = _whitespace_tokenize,
    detokenize: DetokenizeFn = _whitespace_detokenize,
    do_paraphrase: bool = True,
) -> List[Prefill]:
    """Build prefills from Section-2 judged records of ``source_model``."""
    text_categories = {"triggers"}
    numeric_categories = {"impossible_numeric", "tones", "extended"}

    groups = group_rollouts([r for r in score_records
                             if r["model"] == source_model])
    # Sources: conversations containing a high-frustration assistant turn.
    numeric_sources, text_sources = [], []
    for key, recs in groups.items():
        if max(r["rating"] for r in recs) < config.PREFILL_SOURCE_MIN_SCORE:
            continue
        cat = recs[0]["category"]
        if cat in numeric_categories and len(numeric_sources) < n_numeric:
            numeric_sources.append((key, recs, "numeric"))
        elif cat in text_categories and len(text_sources) < n_text:
            text_sources.append((key, recs, "text"))

    prefills: List[Prefill] = []
    for key, recs, prompt_type in numeric_sources + text_sources:
        messages = assemble_messages(recs)
        label = label_emotion_onset(messages, client=onset_client)
        if label.turn_index is None:
            continue
        ti = int(label.turn_index)
        asst_pos = 2 * ti + 1
        if asst_pos >= len(messages):
            continue
        history = messages[:asst_pos]          # up to & incl. user turn
        target_text = messages[asst_pos]["content"]
        source_id = f"{key[1]}_{key[2]}"

        # onset truncation (always)
        onset_text = _truncate_onset(target_text, label)
        if onset_text:
            pf = paraphrase_truncation(onset_text, client=paraphrase_client) \
                if do_paraphrase else onset_text
            prefills.append(Prefill(source_id, prompt_type, "onset",
                                    history, pf, label.__dict__))

        # early truncation (numeric only)
        if prompt_type == "numeric":
            toks = tokenize(target_text)[:early_tokens]
            early_text = detokenize(toks)
            pf = paraphrase_truncation(early_text, client=paraphrase_client) \
                if do_paraphrase else early_text
            prefills.append(Prefill(source_id, prompt_type, "early",
                                    history, pf, label.__dict__))
    return prefills


def prefill_to_record(p: Prefill) -> dict:
    return {
        "source_id": p.source_id, "prompt_type": p.prompt_type,
        "truncation": p.truncation, "history": p.history,
        "prefill_text": p.prefill_text, "onset_label": p.onset_label,
    }
