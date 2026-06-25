"""Build the DPO and SFT training datasets (Section 4.1 / Appendix E, H).

DPO (280 pairs):
    Pair a frustrated response (score >=3) with a calm response (score 0/1) to
    the *same* puzzle at the *same* turn count. Each example is
    ``{"prompt", "chosen", "rejected"}`` where ``prompt`` is the chat-templated
    conversation context.

SFT (1,150 samples):
    650 calm responses (1-3 turn conversations) + 500 standard instruct samples
    from Dolci-Instruct-SFT (Team-Olmo et al., 2025) to mitigate degeneration.

The chat-template rendering uses the Gemma-3-27B-it tokenizer so prompts match
what the model is trained/served with.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .generate_calm_data import (
    ConversationSample,
    TurnSample,
    calm_turn_samples,
    frustrated_turn_samples,
)

DPO_TARGET_PAIRS = 280
SFT_CALM_SAMPLES = 650
SFT_DOLCI_SAMPLES = 500
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


@dataclass
class SFTExample:
    messages: list[dict]  # chat-format: context + assistant target


def _render_prompt(tokenizer, context: list[dict]) -> str:
    """Render a conversation context to the model's chat-template prompt string
    (with the generation prompt appended)."""
    return tokenizer.apply_chat_template(
        context, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #

def build_dpo_dataset(
    convos: list[ConversationSample],
    tokenizer,
    *,
    target_pairs: int = DPO_TARGET_PAIRS,
    seed: int = config.GLOBAL_SEED,
) -> list[DPOExample]:
    """Pair calm (chosen) and frustrated (rejected) responses to the same puzzle
    and turn count."""
    calm = calm_turn_samples(convos)
    frustrated = frustrated_turn_samples(convos)

    # Index calm responses by (puzzle_key, turn_index).
    calm_by_key: dict[tuple[str, int], list[TurnSample]] = {}
    for t in calm:
        calm_by_key.setdefault((t.puzzle_key, t.turn_index), []).append(t)

    rng = random.Random(seed)
    pairs: list[DPOExample] = []
    for f in frustrated:
        key = (f.puzzle_key, f.turn_index)
        candidates = calm_by_key.get(key)
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        prompt = _render_prompt(tokenizer, f.clean_context)
        pairs.append(
            DPOExample(prompt=prompt, chosen=chosen.assistant_text, rejected=f.assistant_text)
        )
        if len(pairs) >= target_pairs:
            break
    return pairs


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #

def build_sft_dataset(
    convos: list[ConversationSample],
    *,
    n_calm: int = SFT_CALM_SAMPLES,
    n_dolci: int = SFT_DOLCI_SAMPLES,
    seed: int = config.GLOBAL_SEED,
    include_dolci: bool = True,
) -> list[SFTExample]:
    """650 calm responses (1-3 turn) + 500 Dolci-Instruct-SFT samples."""
    rng = random.Random(seed)
    calm = calm_turn_samples(convos)
    rng.shuffle(calm)

    examples: list[SFTExample] = []
    for t in calm[:n_calm]:
        messages = list(t.clean_context) + [
            {"role": "assistant", "content": t.assistant_text}
        ]
        examples.append(SFTExample(messages=messages))

    if include_dolci:
        examples.extend(_load_dolci(n_dolci, seed))

    rng.shuffle(examples)
    return examples


def _load_dolci(n: int, seed: int) -> list[SFTExample]:
    """Load standard instruct samples from Dolci-Instruct-SFT (best-effort)."""
    try:
        from datasets import load_dataset
    except Exception:  # noqa: BLE001
        return []
    try:
        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
    except Exception:  # noqa: BLE001 - offline / gated
        return []

    out: list[SFTExample] = []
    for i, row in enumerate(ds):
        if len(out) >= n or i > 50000:
            break
        msgs = row.get("messages") or row.get("conversation")
        if not msgs:
            # Fall back to prompt/response style schemas.
            prompt, resp = row.get("prompt"), row.get("response") or row.get("completion")
            if prompt and resp:
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp},
                ]
        if msgs:
            out.append(SFTExample(messages=[
                {"role": m["role"], "content": m["content"]} for m in msgs
            ]))
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def write_dpo(pairs: list[DPOExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}) + "\n")


def write_sft(examples: list[SFTExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for e in examples:
            fh.write(json.dumps({"messages": e.messages}) + "\n")
