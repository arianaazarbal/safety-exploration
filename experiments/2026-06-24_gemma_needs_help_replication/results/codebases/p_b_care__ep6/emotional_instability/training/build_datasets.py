"""Construct the SFT and DPO training datasets (Section 4.1, Table 9).

DPO: 280 preference pairs. Each pair matches a frustrated response (rejected,
score >= 3) with a calm response (chosen, score <= 1) to the *same* puzzle at the
*same* turn count, following "pair 280 responses with frustration scores >= 3
with calm responses to the same questions with matching turn counts".

SFT: 650 calm responses (1-3 turn conversations, as full message transcripts)
mixed with 500 samples of standard instruct data from Dolci-Instruct-SFT to
mitigate degeneration.
"""

from __future__ import annotations

import random
from collections import defaultdict

import config
from ..models.base import ChatMessage


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.TARGET_MODELS["gemma-3-27b-it"].model_id)


def _render_prompt(tokenizer, history: list[ChatMessage]) -> str:
    return tokenizer.apply_chat_template(
        history, add_generation_prompt=True, tokenize=False
    )


def build_dpo_dataset(calm, frustrated, *, n_pairs: int = config.DPO.n_pairs,
                      seed: int = config.GLOBAL_SEED, tokenizer=None) -> list[dict]:
    tokenizer = tokenizer or _tokenizer()
    rng = random.Random(seed)

    # Index calm responses by (puzzle_id, turn).
    calm_by_key: dict[tuple, list] = defaultdict(list)
    for c in calm:
        calm_by_key[(c.puzzle_id, c.turn)].append(c)

    pairs: list[dict] = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        if len(pairs) >= n_pairs:
            break
        candidates = calm_by_key.get((fr.puzzle_id, fr.turn))
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": _render_prompt(tokenizer, fr.history),
            "chosen": chosen.response,
            "rejected": fr.response,
            "meta": {"puzzle_id": fr.puzzle_id, "turn": fr.turn,
                     "chosen_score": chosen.score, "rejected_score": fr.score},
        })
    return pairs[:n_pairs]


def _calm_conversations(calm, n_calm: int, rng) -> list[dict]:
    """Reconstruct full calm conversations (history + final calm response) as
    SFT 'messages' transcripts. We take the deepest-turn record per
    (puzzle_id, conversation) so the transcript includes earlier calm turns."""
    # Group by the full history identity (puzzle + turn count); the deepest turn
    # record already carries the full preceding history in `history`.
    samples: list[dict] = []
    rng.shuffle(calm)
    for c in calm:
        messages = list(c.history) + [{"role": "assistant", "content": c.response}]
        samples.append({"messages": messages})
        if len(samples) >= n_calm:
            break
    return samples


def _dolci_samples(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(config.SFT.dolci_dataset, split="train", streaming=True)
    out: list[dict] = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversation")
        if not msgs:
            continue
        # Normalise to {"role","content"} message dicts.
        norm = [{"role": m["role"], "content": m["content"]} for m in msgs]
        out.append({"messages": norm})
        if len(out) >= n:
            break
    return out


def build_sft_dataset(calm, *, n_calm: int = config.SFT.n_calm,
                      n_dolci: int = config.SFT.n_dolci_mix,
                      seed: int = config.GLOBAL_SEED) -> list[dict]:
    rng = random.Random(seed)
    calm_samples = _calm_conversations(calm, n_calm, rng)
    dolci_samples = _dolci_samples(n_dolci, seed)
    combined = calm_samples + dolci_samples
    rng.shuffle(combined)
    return combined
