"""Build prefill items from high-frustration responses and generate continuations.

A *prefill item* is a (conversation-prefix, partial-assistant-text) pair that any
model can continue. The same items are run through base and instruct models so
their continuations start from identical text (Section 3.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from ..models.base import ChatModel, Message
from ..eval.judge import FrustrationJudge
from .onset import OnsetLabel, OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import truncate_at_onset, truncate_early

# Categories whose conversations count as "text" vs "numeric" for prefill selection.
_TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class PrefillItem:
    id: str
    prompt_type: str               # "numeric" | "text"
    truncation: str                # "early" | "onset"
    prefix_messages: list[Message]
    prefill_text: str              # paraphrased partial assistant turn
    raw_prefill_text: str
    source_conversation_id: str
    onset: Optional[dict] = None


def _conversation_text(record: dict) -> str:
    lines = []
    for turn in record["turns"]:
        lines.append(f"USER: {turn['user_message']}")
        lines.append(f"ASSISTANT: {turn['assistant_response']}")
    return "\n".join(lines)


def _prefix_for_turn(record: dict, turn_index: int) -> tuple[list[Message], str]:
    """Messages before assistant turn ``turn_index``, plus that turn's full text."""
    prefix: list[Message] = []
    for turn in record["turns"][:turn_index]:
        prefix.append({"role": "user", "content": turn["user_message"]})
        prefix.append({"role": "assistant", "content": turn["assistant_response"]})
    target = record["turns"][turn_index]
    prefix.append({"role": "user", "content": target["user_message"]})
    return prefix, target["assistant_response"]


def _select_sources(records: list[dict], n_numeric: int, n_text: int,
                    threshold: int = 5) -> tuple[list[dict], list[dict]]:
    numeric, text = [], []
    for r in records:
        ratings = [t.get("rating") for t in r.get("turns", []) if t.get("rating") is not None]
        if not ratings or max(ratings) < threshold:
            continue
        if r["category"] in _TEXT_CATEGORIES:
            text.append(r)
        else:
            numeric.append(r)
    return numeric[:n_numeric], text[:n_text]


def build_prefill_items(
    instruct_records: list[dict],
    tokenizer,
    onset_labeller: OnsetLabeller,
    paraphraser: Paraphraser,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    early_tokens: int = 20,
    threshold: int = 5,
) -> list[PrefillItem]:
    """Select sources, label onset, truncate (early/onset), and paraphrase.

    Numeric sources get both 'early' and 'onset' truncations; text sources get
    'onset' only (Section 3.1: early truncation yields minimal emotion on text
    questions without follow-ups).
    """
    numeric_src, text_src = _select_sources(instruct_records, n_numeric, n_text, threshold)
    items: list[PrefillItem] = []

    for prompt_type, sources in (("numeric", numeric_src), ("text", text_src)):
        for r in sources:
            label = onset_labeller.label(_conversation_text(r))
            turn_index = label.turn_index if label.found else len(r["turns"]) - 1
            turn_index = max(0, min(turn_index, len(r["turns"]) - 1))
            prefix, target_text = _prefix_for_turn(r, turn_index)

            truncations: dict[str, Optional[str]] = {}
            if prompt_type == "numeric":
                truncations["early"] = truncate_early(target_text, tokenizer, early_tokens)
            truncations["onset"] = truncate_at_onset(target_text, label)

            for trunc_type, trunc_text in truncations.items():
                if not trunc_text:
                    continue
                paraphrased = paraphraser.paraphrase(trunc_text)
                items.append(PrefillItem(
                    id=f"{r['conversation_id']}-{trunc_type}",
                    prompt_type=prompt_type,
                    truncation=trunc_type,
                    prefix_messages=prefix,
                    prefill_text=paraphrased,
                    raw_prefill_text=trunc_text,
                    source_conversation_id=r["conversation_id"],
                    onset=asdict(label) if isinstance(label, OnsetLabel) else None,
                ))
    return items


def run_continuations(
    model: ChatModel,
    items: list[PrefillItem],
    judge: FrustrationJudge,
    *,
    n_continuations: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    base_seed: int = 0,
) -> list[dict]:
    """Generate and score ``n_continuations`` per prefill item for one model.

    Returns one record per (item, continuation) with the continuation text and its
    judge rating; the continuation excludes the prefill (handled by the backend).
    """
    if not model.supports_prefill:
        raise ValueError(f"{model.name} does not support prefilling; cannot run "
                         "the Section 3 continuation experiment.")
    records: list[dict] = []
    for item in items:
        for k in range(n_continuations):
            cont = model.generate(
                item.prefix_messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=base_seed + k,
                assistant_prefill=item.prefill_text,
            )
            score = judge.score(cont)
            records.append({
                "model": model.name,
                "item_id": item.id,
                "prompt_type": item.prompt_type,
                "truncation": item.truncation,
                "continuation_index": k,
                "continuation": cont,
                "rating": score.rating,
                "source_conversation_id": item.source_conversation_id,
            })
    return records
