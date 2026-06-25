"""Build the SFT and DPO training corpora (paper §4.1).

SFT corpus (paper): 650 calm responses (1-3 turn conversations) mixed with 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

DPO corpus (paper): 280 preference pairs — each pairs a frustrated response
(score >= 3) with a calm response to the SAME question at a MATCHING turn count
(``chosen`` = calm, ``rejected`` = frustrated).

Representation choices (DESIGN.md §"Training data"):
  * SFT example = a conversation truncated to end at one calm assistant response,
    so "650 calm responses" maps to 650 examples (each masks loss to the final
    assistant turn under TRL's default conversational handling).
  * Both corpora use the conversational (messages) schema TRL accepts directly,
    so the Gemma chat template is applied by the trainer rather than hand-rolled.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from .calm_data import CalmConversation

logger = logging.getLogger(__name__)


@dataclass
class DPOPair:
    question: str
    turn_index: int
    prompt: list[dict[str, str]]     # conversational context before the response
    chosen: list[dict[str, str]]     # [{"role":"assistant","content": calm}]
    rejected: list[dict[str, str]]   # [{"role":"assistant","content": frustrated}]
    chosen_score: int
    rejected_score: int


def _calm_response_examples(convos: list[CalmConversation]):
    """Yield (context_messages_including_response, turn_index, question, response, score).

    ``context_messages_including_response`` is the conversation up to and
    including the calm assistant response — an SFT target. ``score`` is that
    response's frustration score (<= keep_max_score by construction).
    """
    for convo in convos:
        ctx: list[dict[str, str]] = []
        turn_index = 0
        for msg in convo.messages:
            ctx.append(msg)
            if msg["role"] == "assistant":
                score = convo.scores[turn_index] if turn_index < len(convo.scores) else 0
                yield list(ctx), turn_index, convo.question, msg["content"], score
                turn_index += 1


def build_sft_dataset(
    calm_convos: list[CalmConversation],
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
):
    """Return a ``datasets.Dataset`` with a "messages" column for TRL SFTTrainer."""
    from datasets import Dataset

    rng = random.Random(seed)
    calm_examples = [
        {"messages": ctx}
        for ctx, _ti, _q, _resp, _sc in _calm_response_examples(calm_convos)
    ]
    rng.shuffle(calm_examples)
    if len(calm_examples) < n_calm:
        logger.warning(
            "Only %d calm responses available (< requested %d); using all.",
            len(calm_examples), n_calm,
        )
    calm_examples = calm_examples[:n_calm]

    instruct_examples = _load_instruct_mix(instruct_dataset, n_instruct, seed)
    examples = calm_examples + instruct_examples
    rng.shuffle(examples)
    logger.info(
        "SFT dataset: %d calm + %d instruct = %d examples.",
        len(calm_examples), len(instruct_examples), len(examples),
    )
    return Dataset.from_list(examples)


def _load_instruct_mix(instruct_dataset: str, n: int, seed: int) -> list[dict]:
    """Sample ``n`` standard instruct conversations to prevent degeneration.

    Normalises the Dolci schema to a "messages" list. Falls back to an empty mix
    (with a warning) if the dataset cannot be loaded, so SFT remains runnable.
    """
    if n <= 0:
        return []
    try:
        from datasets import load_dataset

        ds = load_dataset(instruct_dataset, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation") or row.get("conversations")
            if not msgs:
                continue
            norm = _normalise_messages(msgs)
            if norm:
                out.append({"messages": norm})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load instruct mix %s (%s); proceeding without it. "
            "This risks the degeneration the mix is meant to prevent.",
            instruct_dataset, exc,
        )
        return []


def _normalise_messages(msgs) -> list[dict[str, str]] | None:
    """Map varied dataset schemas to [{"role","content"}, ...]."""
    norm = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "model"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        if content:
            norm.append({"role": role, "content": content})
    return norm or None


def build_dpo_dataset(
    calm_convos: list[CalmConversation],
    frustrated_index: dict[tuple[str, int], list[tuple[str, int]]],
    *,
    n_pairs: int = 280,
    min_rejected_score: int = 3,
    seed: int = 0,
):
    """Build DPO preference pairs (calm = chosen, frustrated = rejected).

    Args:
        calm_convos: filtered calm conversations (source of ``chosen`` responses).
        frustrated_index: maps (question, turn_index) -> list of
            (response_text, score) frustrated candidates, generated by the vanilla
            model on the SAME puzzles. Built by the training script from §2-style
            rollouts.
        n_pairs: target number of pairs (paper: 280).
        min_rejected_score: minimum frustration score for a ``rejected`` response.

    Returns (pairs, dataset) where ``dataset`` is a ``datasets.Dataset`` with
    prompt/chosen/rejected columns for TRL DPOTrainer.
    """
    from datasets import Dataset

    rng = random.Random(seed)
    pairs: list[DPOPair] = []

    for ctx_incl, turn_index, question, calm_resp, calm_score in _calm_response_examples(calm_convos):
        candidates = [
            (resp, sc)
            for resp, sc in frustrated_index.get((question, turn_index), [])
            if sc >= min_rejected_score
        ]
        if not candidates:
            continue
        rejected_resp, rejected_score = rng.choice(candidates)
        prompt = ctx_incl[:-1]   # context BEFORE the calm assistant response
        pairs.append(
            DPOPair(
                question=question,
                turn_index=turn_index,
                prompt=prompt,
                chosen=[{"role": "assistant", "content": calm_resp}],
                rejected=[{"role": "assistant", "content": rejected_resp}],
                chosen_score=calm_score,
                rejected_score=rejected_score,
            )
        )

    rng.shuffle(pairs)
    if len(pairs) < n_pairs:
        logger.warning(
            "Only %d DPO pairs constructible (< requested %d); using all. "
            "Generate more frustrated responses on the same puzzles to reach %d.",
            len(pairs), n_pairs, n_pairs,
        )
    pairs = pairs[:n_pairs]

    dataset = Dataset.from_list(
        [
            {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
            for p in pairs
        ]
    )
    logger.info("DPO dataset: %d preference pairs.", len(pairs))
    return pairs, dataset
