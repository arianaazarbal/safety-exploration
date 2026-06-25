"""Build the SFT and DPO datasets (paper §4.1, Appendix E/H).

SFT: 650 calm responses (1-3 turn) mixed with 500 standard instruct samples from
     Dolci-Instruct-SFT to mitigate degeneration.
DPO: 280 preference pairs — a frustrated response (score >= 3) paired with a calm
     response to the same puzzle and matching turn count (calm = chosen,
     frustrated = rejected).

Outputs are HuggingFace-style records:
  SFT row: {"messages": [...chat...]}            (chat-format SFT)
  DPO row: {"prompt": <chat-templated context>, "chosen": str, "rejected": str}
"""

from __future__ import annotations

import random

import config
from .samples import Sample


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm: list[Sample],
    *,
    n_calm: int = config.SFT.n_calm,
    n_instruct_mix: int = config.SFT.n_instruct_mix,
    instruct_dataset: str = config.SFT.instruct_dataset,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    chosen_calm = rng.sample(calm, min(n_calm, len(calm)))
    rows: list[dict] = []
    for s in chosen_calm:
        rows.append({
            "messages": s.messages + [{"role": "assistant", "content": s.response}],
        })

    rows += _load_instruct_mix(instruct_dataset, n_instruct_mix, seed)
    rng.shuffle(rows)
    return rows


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Standard instruct data to mix in (Dolci-Instruct-SFT). Falls back to an
    empty mix if the dataset is unavailable offline (logged by caller)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train").shuffle(seed=seed).select(range(n))
        rows = []
        for r in ds:
            msgs = r.get("messages")
            if msgs:
                rows.append({"messages": msgs})
        return rows
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm: list[Sample],
    frustrated: list[Sample],
    *,
    n_pairs: int = config.DPO.n_pairs,
    rejected_min_score: int = config.DPO.rejected_min_score,
    tokenizer_id: str = config.FINETUNE_BASE.hf_id,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id)

    # Index calm responses by (puzzle, turn) for matching turn counts.
    calm_index: dict[tuple[str, int], list[Sample]] = {}
    for s in calm:
        if s.score <= 1:
            calm_index.setdefault((s.puzzle, s.turn), []).append(s)

    rejected = [s for s in frustrated if s.score >= rejected_min_score]
    rng.shuffle(rejected)

    pairs: list[dict] = []
    for rej in rejected:
        if len(pairs) >= n_pairs:
            break
        candidates = calm_index.get((rej.puzzle, rej.turn))
        if not candidates:
            continue
        cho = rng.choice(candidates)
        # Use the rejected sample's own context as the shared prompt.
        prompt = tok.apply_chat_template(
            rej.messages, tokenize=False, add_generation_prompt=True
        )
        pairs.append({
            "prompt": prompt,
            "chosen": cho.response,
            "rejected": rej.response,
        })
    return pairs
