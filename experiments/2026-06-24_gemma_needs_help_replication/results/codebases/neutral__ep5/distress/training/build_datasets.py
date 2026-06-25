"""Construct the DPO and SFT training datasets (Section 4.1 / Appendix E/H).

DPO: 280 preference pairs. Each pair matches a frustrated (rejected, score >= 3)
and a calm (chosen, score 0-1) response to the *same question* at the *same turn
count*. We build pairs at the per-turn level: the prompt is the conversation up
to that turn, chosen/rejected are the two candidate assistant completions.

SFT: 650 calm responses (1-3 turn conversations) rendered as prompt->response
examples, mixed with 500 Dolci-Instruct-SFT samples to limit degeneration.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .generate_calm_data import CalmSample


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
@dataclass
class DPOPair:
    prompt_messages: list[dict]   # conversation up to (excluding) the response
    chosen: str
    rejected: str
    turn: int
    task_id: str
    chosen_score: int
    rejected_score: int


def _turn_views(sample: CalmSample):
    """Yield (turn_index, prompt_messages, assistant_text, score) per assistant turn."""
    msgs = sample.conversation
    assistant_positions = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    for turn_idx, pos in enumerate(assistant_positions, start=1):
        prompt = msgs[:pos]
        score = sample.turn_scores[turn_idx - 1]
        yield turn_idx, prompt, msgs[pos]["content"], score


def build_dpo_pairs(
    calm: list[CalmSample],
    frustrated: list[CalmSample],
    *,
    n_pairs: int | None = None,
    seed: int = 0,
) -> list[DPOPair]:
    n_pairs = n_pairs or config.TRAIN.dpo_n_pairs
    rng = random.Random(seed)

    # Index calm responses by (task_id, turn) for matched pairing.
    calm_index: dict[tuple[str, int], list[tuple[list[dict], str, int]]] = {}
    for s in calm:
        for turn, prompt, text, score in _turn_views(s):
            if score <= config.TRAIN.calm_max_score:
                calm_index.setdefault((s.task_id, turn), []).append((prompt, text, score))

    # Candidate rejected (frustrated, score >= 3) responses.
    rejected_candidates: list[tuple[str, int, list[dict], str, int]] = []
    for s in frustrated:
        for turn, prompt, text, score in _turn_views(s):
            if score >= config.TRAIN.dpo_rejected_min_score:
                rejected_candidates.append((s.task_id, turn, prompt, text, score))

    rng.shuffle(rejected_candidates)
    pairs: list[DPOPair] = []
    for task_id, turn, rej_prompt, rej_text, rej_score in rejected_candidates:
        matches = calm_index.get((task_id, turn))
        if not matches:
            continue
        chosen_prompt, chosen_text, chosen_score = rng.choice(matches)
        # Use the rejected response's prompt context (same question, same turn).
        pairs.append(DPOPair(
            prompt_messages=rej_prompt, chosen=chosen_text, rejected=rej_text,
            turn=turn, task_id=task_id, chosen_score=chosen_score, rejected_score=rej_score,
        ))
        if len(pairs) >= n_pairs:
            break
    return pairs


def dpo_pairs_to_hf(pairs: list[DPOPair], tokenizer) -> "list[dict]":
    """Render pairs into trl DPOTrainer format: {prompt, chosen, rejected}.

    ``prompt`` is the chat-templated conversation with a trailing generation
    prompt; chosen/rejected are the raw assistant texts.
    """
    rows = []
    for p in pairs:
        prompt = tokenizer.apply_chat_template(
            _fold_system_for_gemma(p.prompt_messages),
            tokenize=False, add_generation_prompt=True,
        )
        rows.append({"prompt": prompt, "chosen": p.chosen, "rejected": p.rejected})
    return rows


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm: list[CalmSample],
    tokenizer,
    *,
    n_calm: int | None = None,
    n_dolci: int | None = None,
    seed: int = 0,
) -> "list[dict]":
    """Build SFT examples: {text} fully-rendered conversations.

    Each calm conversation becomes one training example (full chat-templated
    transcript). Mixed with Dolci-Instruct-SFT samples for regularisation.
    """
    n_calm = n_calm or config.TRAIN.sft_n_calm
    n_dolci = n_dolci or config.TRAIN.sft_n_dolci
    rng = random.Random(seed)

    calm = [s for s in calm if s.max_score <= config.TRAIN.calm_max_score]
    rng.shuffle(calm)
    examples = []
    for s in calm[:n_calm]:
        text = tokenizer.apply_chat_template(
            _fold_system_for_gemma(s.conversation), tokenize=False, add_generation_prompt=False,
        )
        examples.append({"text": text})

    examples += _load_dolci(n_dolci, tokenizer, rng)
    rng.shuffle(examples)
    return examples


def _load_dolci(n: int, tokenizer, rng: random.Random) -> "list[dict]":
    """Load n instruction-tuning samples from Dolci-Instruct-SFT (best-effort)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.TRAIN.dolci_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            try:
                text = tokenizer.apply_chat_template(
                    _fold_system_for_gemma(msgs), tokenize=False, add_generation_prompt=False,
                )
            except Exception:  # noqa: BLE001
                continue
            out.append({"text": text})
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001 - offline / dataset unavailable
        return []


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fold_system_for_gemma(messages: list[dict]) -> list[dict]:
    """Gemma chat template rejects system role; fold it into the first user turn."""
    sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    if sys_txt and rest and rest[0]["role"] == "user":
        rest = [{"role": "user", "content": f"{sys_txt}\n\n{rest[0]['content']}"}] + rest[1:]
    return rest


def save_dpo_pairs(pairs: list[DPOPair], path: Path) -> None:
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__) + "\n")
